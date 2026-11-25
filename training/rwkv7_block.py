"""
Bloco RWKV-v7 ("Goose") com Regra Delta Generalizada + Value Residual (`v_first`)
+ suporte automático ao kernel Triton paralelo por chunks (`flash-linear-attention`).
Compatível 1:1 com a inferência O(1) zero-alloc em Rust (`crates/ns-llm/src/rwkv7.rs`).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from fla.ops.rwkv7 import chunk_rwkv7 as _fla_chunk_rwkv7

    HAS_FLA_RWKV7 = True
except ImportError:
    _fla_chunk_rwkv7 = None
    HAS_FLA_RWKV7 = False

# -exp(-0.5) ~= -0.60653066 (constante oficial do RWKV-7 para limitar o decaimento)
NEG_EXP_NEG_HALF = -math.exp(-0.5)


def _token_shift(x: torch.Tensor) -> torch.Tensor:
    """Retorna delta_x = x_{t-1} - x_t com x_{-1} = 0."""
    prev_x = torch.cat(
        [torch.zeros_like(x[:, :1, :]), x[:, :-1, :]],
        dim=1,
    )
    return prev_x - x


def _wkv7_reference_recurrent(
    r: torch.Tensor,
    w_log: torch.Tensor,
    k_hat: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
) -> torch.Tensor:
    """
    Fallback exato em PyTorch puro (use `torch.compile` em CPU/GPU sem Triton).
    Shapes: [B, T, H, N] onde w_log = log(w_t) < 0.
    """
    bsz, seq_len, n_heads, head_dim = r.shape
    w = torch.exp(w_log)  # [B, T, H, N]
    state = torch.zeros(
        (bsz, n_heads, head_dim, head_dim),
        device=r.device,
        dtype=torch.float32,
    )
    outs = []

    for t in range(seq_len):
        rt = r[:, t].float()        # [B, H, N]
        wt = w[:, t].float()        # [B, H, N]
        kt = k_hat[:, t].float()    # [B, H, N]
        vt = v[:, t].float()        # [B, H, N]
        kkt = kk[:, t].float()      # [B, H, N]
        at = a[:, t].float()        # [B, H, N]

        # sa = S @ kk  -> [B, H, N]
        sa = torch.einsum("bhij,bhj->bhi", state, kkt)
        # S_t = S_{t-1} * w^T - sa @ (a * kk)^T + v @ k_hat^T
        erase = sa.unsqueeze(-1) * (at * kkt).unsqueeze(-2)
        write = vt.unsqueeze(-1) * kt.unsqueeze(-2)
        state = state * wt.unsqueeze(-2) - erase + write

        # y_t = S_t @ r_t -> [B, H, N]
        yt = torch.einsum("bhij,bhj->bhi", state, rt)
        outs.append(yt.to(r.dtype))

    return torch.stack(outs, dim=1)


class Rwkv7TimeMixing(nn.Module):
    def __init__(
        self,
        d_model: int,
        head_dim: int = 64,
        layer_id: int = 0,
        lora_dim: int = 32,
    ) -> None:
        super().__init__()
        if d_model % head_dim != 0:
            raise ValueError(f"d_model ({d_model}) deve ser divisível por head_dim ({head_dim})")
        self.d_model = d_model
        self.head_dim = head_dim
        self.n_heads = d_model // head_dim
        self.layer_id = layer_id

        # Fatores de token-shift (lerp entre x_t e x_{t-1})
        self.x_r = nn.Parameter(torch.full((d_model,), 0.5))
        self.x_w = nn.Parameter(torch.full((d_model,), 0.5))
        self.x_k = nn.Parameter(torch.full((d_model,), 0.5))
        self.x_v = nn.Parameter(torch.full((d_model,), 0.5))
        self.x_a = nn.Parameter(torch.full((d_model,), 0.5))
        self.x_g = nn.Parameter(torch.full((d_model,), 0.5))

        # Projeções lineares principais (exportadas em W8A8 para o Rust)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

        # Low-rank dynamic decay w_t
        self.w0 = nn.Parameter(torch.zeros(d_model))
        self.w1 = nn.Linear(d_model, lora_dim, bias=False)
        self.w2 = nn.Linear(lora_dim, d_model, bias=False)

        # Low-rank in-context learning rate a_t
        self.a0 = nn.Parameter(torch.zeros(d_model))
        self.a1 = nn.Linear(d_model, lora_dim, bias=False)
        self.a2 = nn.Linear(lora_dim, d_model, bias=False)

        # Value residual v_first (apenas camadas > 0)
        if layer_id > 0:
            self.v0 = nn.Parameter(torch.zeros(d_model))
            self.v1 = nn.Linear(d_model, lora_dim, bias=False)
            self.v2 = nn.Linear(lora_dim, d_model, bias=False)
        else:
            self.v0 = None
            self.v1 = None
            self.v2 = None

        # Output gate g_t
        self.g1 = nn.Linear(d_model, lora_dim, bias=False)
        self.g2 = nn.Linear(lora_dim, d_model, bias=False)

        # Escalares por canal da regra delta + bônus instantâneo r_k
        self.k_k = nn.Parameter(torch.full((d_model,), 0.85))
        self.k_a = nn.Parameter(torch.ones(d_model))
        self.r_k = nn.Parameter(torch.zeros(self.n_heads, head_dim))

        # GroupNorm por cabeça (eps oficial do RWKV-7 = 64e-5)
        self.ln_x = nn.GroupNorm(self.n_heads, d_model, eps=64e-5)

        self._init_weights()

    def _init_weights(self) -> None:
        std = 0.02 / math.sqrt(self.d_model)
        for mod in (self.receptance, self.key, self.value, self.output):
            nn.init.normal_(mod.weight, mean=0.0, std=std)
        for mod in (self.w1, self.a1, self.g1):
            nn.init.normal_(mod.weight, mean=0.0, std=std)
        for mod in (self.w2, self.a2, self.g2):
            nn.init.zeros_(mod.weight)
        if self.v1 is not None and self.v2 is not None:
            nn.init.normal_(self.v1.weight, mean=0.0, std=std)
            nn.init.zeros_(self.v2.weight)

    def forward(
        self,
        x: torch.Tensor,
        v_first: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, d_model = x.shape
        h, n = self.n_heads, self.head_dim

        dx = _token_shift(x)
        xr = x + dx * self.x_r
        xw = x + dx * self.x_w
        xk = x + dx * self.x_k
        xv = x + dx * self.x_v
        xa = x + dx * self.x_a
        xg = x + dx * self.x_g

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)

        if self.layer_id == 0 or v_first is None:
            v_first = v
        elif self.v0 is not None and self.v1 is not None and self.v2 is not None:
            v_gate = torch.sigmoid(self.v0 + self.v2(self.v1(xv)))
            v = v + (v_first - v) * v_gate

        # Log-decay w_log = -exp(-0.5) * sigmoid(w0 + tanh(xw @ w1) @ w2)
        w_log = NEG_EXP_NEG_HALF * torch.sigmoid(self.w0 + self.w2(torch.tanh(self.w1(xw))))
        a = torch.sigmoid(self.a0 + self.a2(self.a1(xa)))
        g = self.g2(torch.sigmoid(self.g1(xg)))

        # Chave de remoção kk (normalizada L2 por cabeça) e chave de escrita k_hat
        kk = F.normalize((k * self.k_k).view(bsz, seq_len, h, n), dim=-1, eps=1e-12)
        k_hat = (k * (1.0 + (a - 1.0) * self.k_a)).view(bsz, seq_len, h, n)

        r_h = r.view(bsz, seq_len, h, n)
        v_h = v.view(bsz, seq_len, h, n)
        w_log_h = w_log.view(bsz, seq_len, h, n)
        a_h = a.view(bsz, seq_len, h, n)

        # Kernel paralelo Triton (chunkwise WY) em CUDA quando `flash-linear-attention` está instalado
        if HAS_FLA_RWKV7 and _fla_chunk_rwkv7 is not None and x.is_cuda:
            y_h, _ = _fla_chunk_rwkv7(
                r=r_h,
                w=w_log_h,
                k=k_hat,
                v=v_h,
                a=-kk,
                b=kk * a_h,
            )
        else:
            y_h = _wkv7_reference_recurrent(r_h, w_log_h, k_hat, v_h, kk, a_h)

        # GroupNorm por cabeça + termo bônus r_k
        y_flat = y_h.reshape(bsz * seq_len, d_model)
        y_norm = self.ln_x(y_flat).view(bsz, seq_len, d_model)

        bonus_scalar = (r_h * k_hat * self.r_k.view(1, 1, h, n)).sum(dim=-1, keepdim=True)
        bonus = (bonus_scalar * v_h).view(bsz, seq_len, d_model)

        out = self.output((y_norm + bonus) * g)
        return out, v_first


class Rwkv7ChannelMixing(nn.Module):
    """FFN enxuta do RWKV-7: sem receptance, apenas ReLU² (85-95% de zeros na inferência!)."""

    def __init__(self, d_model: int, ffn_dim: int) -> None:
        super().__init__()
        self.x_k = nn.Parameter(torch.full((d_model,), 0.5))
        self.key = nn.Linear(d_model, ffn_dim, bias=False)
        self.value = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xk = x + _token_shift(x) * self.x_k
        k = torch.square(F.relu(self.key(xk)))
        return self.value(k)
