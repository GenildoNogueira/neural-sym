"""
Importa o checkpoint oficial BlinkDL RWKV-7 G1 (ex.: rwkv7-g1d-0.1b-20260129-ctx8192.pth),
acopla a Dual-Head ISA + Entity Pointer (ou Copy Gate em modo JSON), executa pós-treino SFT
no verified_dataset.jsonl (com continuação de estado O(1) e kernel Triton `fla` em CUDA)
e exporta para `crates/ns-llm/src/rwkv7.rs`.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

try:
    from fla.ops.rwkv7 import chunk_rwkv7 as _fla_chunk_rwkv7

    HAS_FLA_RWKV7 = True
except ImportError:
    _fla_chunk_rwkv7 = None
    HAS_FLA_RWKV7 = False

NEG_EXP_NEG_HALF = -math.exp(-0.5)

LINEAR_WEIGHT_SUFFIXES = (
    ".tm.receptance.weight",
    ".tm.key.weight",
    ".tm.value.weight",
    ".tm.output.weight",
    ".tm.w1.weight",
    ".tm.w2.weight",
    ".tm.a1.weight",
    ".tm.a2.weight",
    ".tm.v1.weight",
    ".tm.v2.weight",
    ".tm.g1.weight",
    ".tm.g2.weight",
    ".cm.key.weight",
    ".cm.value.weight",
    "head.weight",
    "copy_q.weight",
    "copy_k.weight",
)

RE_SYNTH_FACT_ID = re.compile(r"\bfact-(?:preco|taxa|upgrade)-\d{3}\b")
RE_SYNTH_ENTITY = re.compile(r"\bplano_\d{3}\b")
AUG_ENTITY_PREFIXES = ("plano", "conta", "modulo", "pacote", "servico", "contrato")
AUG_ENTITY_SUFFIXES = ("pro", "plus", "max", "alpha", "prime", "cloud", "enterprise")
AUG_FACT_PREFIXES = ("fact", "doc", "reg", "ref", "item", "cl")
AUG_FACT_TAGS = ("valor", "custo", "regra", "tarifa", "chave", "base")

ISA_OP_NAMES = (
    "query_fact",      # 0 -> Op::QueryFact
    "search_facts",    # 1 -> Op::SearchFacts
    "traverse_graph",  # 2 -> Op::TraverseGraph
    "eval_math",       # 3 -> Op::EvalMath
    "propose_answer",  # 4 -> Op::ProposeAnswer
    "search_context",  # 5 -> Op::SearchContext
    "read_spans",      # 6 -> Op::ReadSpans
    "check_logic",     # 7 -> Op::CheckLogic
    "run_wasm_check",  # 8 -> Op::RunWasmCheck
)
ISA_OP_TO_IDX = {name: i for i, name in enumerate(ISA_OP_NAMES)}

ISA_OP_SEED_WORDS = (
    "query lookup fact entity relation",
    "search find facts database",
    "traverse graph hops links",
    "evaluate calculate math decimal",
    "propose final answer cite",
    "search document context clause",
    "read spans clauses compare",
    "check logic smt verify",
    "run wasm sandbox code",
)

CTL_END = 0
CTL_LIT_OPEN = 1
CTL_LIT_CLOSE = 2

LayerState = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class Rwkv7Config:
    arch: str = "rwkv7"
    vocab_size: int = 65536
    d_model: int = 768
    n_layers: int = 12
    head_dim: int = 64
    ffn_dim: int = 3072
    decay_lora_dim: int = 64
    a_lora_dim: int = 64
    v_lora_dim: int = 32
    gate_lora_dim: int = 128
    bos_token_id: int = 0
    eos_token_id: int = 0
    layer_norm_eps: float = 1e-5
    group_norm_eps: float = 64e-5
    vocab_file: str = "vocab.bin"
    copy_dim: int = 64
    copy_max_len: int = 512
    isa_ptr_dim: int = 0
    isa_num_vocab: int = 64


class RwkvWorldTokenizer:
    """
    Lê `rwkv_vocab_v20230424.txt`, executa greedy longest-match Trie sobre bytes
    (100% compatível com `TokTrie::greedy_tokenize` no Rust) e exporta `vocab.bin`.
    """

    def __init__(self, vocab_txt: Path, vocab_size: int = 65536) -> None:
        self.vocab_size = vocab_size
        self.idx2bytes: list[bytes] = [b""] * vocab_size
        self.trie: dict[int, dict] = {}

        with vocab_txt.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                first_sp = line.index(" ")
                last_sp = line.rindex(" ")
                idx = int(line[:first_sp])
                raw_literal = line[first_sp + 1 : last_sp]
                parsed = ast.literal_eval(raw_literal)
                token_bytes = parsed.encode("utf-8") if isinstance(parsed, str) else bytes(parsed)
                if 0 < idx < vocab_size:
                    self.idx2bytes[idx] = token_bytes
                    self._insert(token_bytes, idx)

    def _insert(self, token_bytes: bytes, idx: int) -> None:
        node = self.trie
        for b in token_bytes:
            node = node.setdefault(b, {})
        node[-1] = idx

    def encode(self, text: str) -> list[int]:
        data = text.encode("utf-8")
        n = len(data)
        pos = 0
        ids: list[int] = []
        while pos < n:
            node = self.trie
            best_idx = -1
            best_end = pos
            cur = pos
            while cur < n and data[cur] in node:
                node = node[data[cur]]
                cur += 1
                if -1 in node:
                    best_idx = node[-1]
                    best_end = cur
            if best_idx < 0:
                raise ValueError(
                    f"byte não coberto no vocabulário RWKV World na pos {pos}: {data[pos]}"
                )
            ids.append(best_idx)
            pos = best_end
        return ids

    def decode(self, ids: list[int]) -> str:
        buf = bytearray()
        for idx in ids:
            if 0 < idx < self.vocab_size:
                buf.extend(self.idx2bytes[idx])
        return buf.decode("utf-8", errors="replace")

    def export_binary(self, out_path: Path) -> None:
        """Formato binário compacto para o Rust: [u32 vocab_size] seguido de ([u16 len][bytes]) * V."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as fh:
            fh.write(struct.pack("<I", self.vocab_size))
            for tok_bytes in self.idx2bytes:
                if len(tok_bytes) > 65535:
                    raise ValueError(f"token excede 65535 bytes: {len(tok_bytes)}")
                fh.write(struct.pack("<H", len(tok_bytes)))
                fh.write(tok_bytes)


def _token_shift(x: torch.Tensor, prev_x: torch.Tensor | None = None) -> torch.Tensor:
    first = torch.zeros_like(x[:, :1, :]) if prev_x is None else prev_x.unsqueeze(1).to(x.dtype)
    shifted = torch.cat([first, x[:, :-1, :]], dim=1)
    return shifted - x


class IsaHeadModule(nn.Module):
    def __init__(self, d_model: int, ptr_dim: int, num_vocab: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.ptr_dim = ptr_dim
        self.num_vocab = num_vocab

        self.op_emb = nn.Embedding(len(ISA_OP_NAMES), d_model)
        self.op_head = nn.Linear(d_model, len(ISA_OP_NAMES), bias=False)
        self.num_emb = nn.Embedding(num_vocab, d_model)
        self.num_head = nn.Linear(d_model, num_vocab, bias=False)
        self.ctl_emb = nn.Embedding(3, d_model)
        self.ctl_head = nn.Linear(d_model, 3, bias=False)

        self.ptr_q = nn.Linear(d_model, ptr_dim, bias=False)
        self.ptr_k = nn.Linear(d_model, ptr_dim, bias=False)
        self.ptr_in = nn.Linear(ptr_dim, d_model, bias=False)
        self.ptr_tag = nn.Parameter(torch.zeros(d_model))

        std = 0.02 / math.sqrt(d_model)
        for mod in (self.op_head, self.num_head, self.ctl_head, self.ptr_q, self.ptr_k, self.ptr_in):
            nn.init.normal_(mod.weight, mean=0.0, std=std)

    @torch.no_grad()
    def semantic_init(self, emb_weight: torch.Tensor, tok: RwkvWorldTokenizer) -> None:
        """Inicializa `op_emb`, `ctl_emb` e `num_emb` com médias dos embeddings pré-treinados."""

        def avg_emb(phrase: str) -> torch.Tensor:
            ids = tok.encode(phrase)
            return emb_weight[ids].mean(dim=0)

        for idx, words in enumerate(ISA_OP_SEED_WORDS):
            self.op_emb.weight[idx].copy_(avg_emb(words))
        for n in range(self.num_vocab):
            self.num_emb.weight[n].copy_(avg_emb(str(n)))
        self.ctl_emb.weight[CTL_END].copy_(avg_emb("end finish stop"))
        self.ctl_emb.weight[CTL_LIT_OPEN].copy_(avg_emb('text literal "'))
        self.ctl_emb.weight[CTL_LIT_CLOSE].copy_(avg_emb('" end literal'))
        self.ptr_tag.copy_(avg_emb("pointer reference slot"))


def _wkv7_recurrent(
    r: torch.Tensor,
    w_log: torch.Tensor,
    k_hat: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    init_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len, n_heads, head_dim = r.shape
    w = torch.exp(w_log)
    state = (
        torch.zeros((bsz, n_heads, head_dim, head_dim), device=r.device, dtype=torch.float32)
        if init_state is None
        else init_state.float().clone()
    )
    outs = []
    for t in range(seq_len):
        rt = r[:, t].float()
        wt = w[:, t].float()
        kt = k_hat[:, t].float()
        vt = v[:, t].float()
        kkt = kk[:, t].float()
        at = a[:, t].float()

        sa = torch.einsum("bhij,bhj->bhi", state, kkt)
        state = (
            state * wt.unsqueeze(-2)
            - sa.unsqueeze(-1) * (at * kkt).unsqueeze(-2)
            + vt.unsqueeze(-1) * kt.unsqueeze(-2)
        )
        yt = torch.einsum("bhij,bhj->bhi", state, rt)
        outs.append(yt.to(r.dtype))
    return torch.stack(outs, dim=1), state


class Rwkv7TimeMixing(nn.Module):
    def __init__(self, cfg: Rwkv7Config, layer_id: int) -> None:
        super().__init__()
        d = cfg.d_model
        self.d_model = d
        self.head_dim = cfg.head_dim
        self.n_heads = d // cfg.head_dim
        self.layer_id = layer_id

        self.x_r = nn.Parameter(torch.zeros(d))
        self.x_w = nn.Parameter(torch.zeros(d))
        self.x_k = nn.Parameter(torch.zeros(d))
        self.x_v = nn.Parameter(torch.zeros(d))
        self.x_a = nn.Parameter(torch.zeros(d))
        self.x_g = nn.Parameter(torch.zeros(d))

        self.receptance = nn.Linear(d, d, bias=False)
        self.key = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        self.output = nn.Linear(d, d, bias=False)

        self.w0 = nn.Parameter(torch.zeros(d))
        self.w1 = nn.Linear(d, cfg.decay_lora_dim, bias=False)
        self.w2 = nn.Linear(cfg.decay_lora_dim, d, bias=False)

        self.a0 = nn.Parameter(torch.zeros(d))
        self.a1 = nn.Linear(d, cfg.a_lora_dim, bias=False)
        self.a2 = nn.Linear(cfg.a_lora_dim, d, bias=False)

        if layer_id > 0:
            self.v0 = nn.Parameter(torch.zeros(d))
            self.v1 = nn.Linear(d, cfg.v_lora_dim, bias=False)
            self.v2 = nn.Linear(cfg.v_lora_dim, d, bias=False)
        else:
            self.v0 = None
            self.v1 = None
            self.v2 = None

        self.g1 = nn.Linear(d, cfg.gate_lora_dim, bias=False)
        self.g2 = nn.Linear(cfg.gate_lora_dim, d, bias=False)

        self.k_k = nn.Parameter(torch.ones(d))
        self.k_a = nn.Parameter(torch.ones(d))
        self.r_k = nn.Parameter(torch.zeros(d))
        self.ln_x = nn.GroupNorm(self.n_heads, d, eps=cfg.group_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        v_first: torch.Tensor | None,
        tm_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        bsz, seq_len, d = x.shape
        h, n = self.n_heads, self.head_dim

        prev_x = None if tm_state is None else tm_state[0]
        init_mat = None if tm_state is None else tm_state[1]
        dx = _token_shift(x, prev_x)
        next_prev_x = x[:, -1, :]

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
            v = v + (v_first - v) * torch.sigmoid(self.v0 + self.v2(self.v1(xv)))

        w_log = NEG_EXP_NEG_HALF * torch.sigmoid(self.w0 + self.w2(torch.tanh(self.w1(xw))))
        a = torch.sigmoid(self.a0 + self.a2(self.a1(xa)))
        g = self.g2(torch.sigmoid(self.g1(xg)))

        kk = F.normalize((k * self.k_k).view(bsz, seq_len, h, n), dim=-1, eps=1e-12)
        k_hat = (k * (1.0 + (a - 1.0) * self.k_a)).view(bsz, seq_len, h, n)

        r_h = r.view(bsz, seq_len, h, n)
        v_h = v.view(bsz, seq_len, h, n)
        w_log_h = w_log.view(bsz, seq_len, h, n)
        a_h = a.view(bsz, seq_len, h, n)

        if HAS_FLA_RWKV7 and _fla_chunk_rwkv7 is not None and x.is_cuda:
            y_h, next_mat = _fla_chunk_rwkv7(
                r=r_h,
                w=w_log_h,
                k=k_hat,
                v=v_h,
                a=-kk,
                b=kk * a_h,
                initial_state=init_mat,
                output_final_state=True,
            )
        else:
            y_h, next_mat = _wkv7_recurrent(r_h, w_log_h, k_hat, v_h, kk, a_h, init_mat)

        y_norm = self.ln_x(y_h.reshape(bsz * seq_len, d)).view(bsz, seq_len, d)
        rk = self.r_k.view(1, 1, h, n)
        bonus = ((r_h * k_hat * rk).sum(dim=-1, keepdim=True) * v_h).view(bsz, seq_len, d)
        return self.output((y_norm + bonus) * g), v_first, (next_prev_x, next_mat)


class Rwkv7ChannelMixing(nn.Module):
    def __init__(self, cfg: Rwkv7Config) -> None:
        super().__init__()
        self.x_k = nn.Parameter(torch.zeros(cfg.d_model))
        self.key = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.value = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        prev_x_cm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xk = x + _token_shift(x, prev_x_cm) * self.x_k
        return self.value(torch.square(F.relu(self.key(xk)))), x[:, -1, :]


class Rwkv7Block(nn.Module):
    def __init__(self, cfg: Rwkv7Config, layer_id: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.tm = Rwkv7TimeMixing(cfg, layer_id)
        self.ln2 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.cm = Rwkv7ChannelMixing(cfg)

    def forward(
        self,
        x: torch.Tensor,
        v_first: torch.Tensor | None,
        layer_state: LayerState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, LayerState]:
        tm_st = None if layer_state is None else (layer_state[0], layer_state[1])
        cm_st = None if layer_state is None else layer_state[2]
        tm_out, v_first, (next_tm_x, next_mat) = self.tm(self.ln1(x), v_first, tm_st)
        x = x + tm_out
        cm_out, next_cm_x = self.cm(self.ln2(x), cm_st)
        x = x + cm_out
        return x, v_first, (next_tm_x, next_mat, next_cm_x)


class NeuroSymRwkv7(nn.Module):
    def __init__(self, cfg: Rwkv7Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.ln_in = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.blocks = nn.ModuleList([Rwkv7Block(cfg, i) for i in range(cfg.n_layers)])
        self.ln_out = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.isa = (
            IsaHeadModule(cfg.d_model, cfg.isa_ptr_dim, cfg.isa_num_vocab)
            if cfg.isa_ptr_dim > 0
            else None
        )

        if cfg.copy_dim > 0:
            self.copy_q = nn.Linear(cfg.d_model, cfg.copy_dim, bias=False)
            self.copy_k = nn.Linear(cfg.d_model, cfg.copy_dim, bias=False)
            self.copy_gate = nn.Linear(cfg.d_model + cfg.copy_dim, 1, bias=True)
            std = 0.02 / math.sqrt(cfg.d_model)
            nn.init.normal_(self.copy_q.weight, mean=0.0, std=std)
            nn.init.normal_(self.copy_k.weight, mean=0.0, std=std)
            nn.init.zeros_(self.copy_gate.weight)
            nn.init.constant_(self.copy_gate.bias, -3.0)
        else:
            self.copy_q = None
            self.copy_k = None
            self.copy_gate = None

    def forward_hidden_embs(
        self,
        input_embs: torch.Tensor,
        states: list[LayerState] | None = None,
    ) -> tuple[torch.Tensor, list[LayerState]]:
        x = self.ln_in(input_embs)
        v_first: torch.Tensor | None = None
        next_states: list[LayerState] = []
        for idx, block in enumerate(self.blocks):
            st_in = None if states is None else states[idx]
            x, v_first, st_out = block(x, v_first, st_in)
            next_states.append(st_out)
        return self.ln_out(x), next_states

    def forward_hidden(self, token_ids: torch.Tensor) -> torch.Tensor:
        h, _ = self.forward_hidden_embs(self.emb(token_ids))
        return h

    def forward(
        self,
        token_ids: torch.Tensor,
        src_valid_mask: torch.Tensor | None = None,
        copy_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.forward_hidden(token_ids)
        vocab_logits = self.head(h)

        if (
            self.cfg.copy_dim <= 0
            or self.copy_q is None
            or self.copy_k is None
            or self.copy_gate is None
            or src_valid_mask is None
            or copy_token_ids is None
        ):
            return F.log_softmax(vocab_logits, dim=-1)

        bsz, seq_len, _ = h.shape
        d_c = self.cfg.copy_dim
        q = self.copy_q(h)
        k = self.copy_k(h)

        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d_c)
        mask_3d = src_valid_mask.unsqueeze(1)
        scores = scores.masked_fill(~mask_3d, -1e9)
        attn = F.softmax(scores, dim=-1) * mask_3d.to(q.dtype)

        ctx = torch.bmm(attn, k)
        p_copy = torch.sigmoid(self.copy_gate(torch.cat([h, ctx], dim=-1)))
        has_src = src_valid_mask.any(dim=1).view(bsz, 1, 1).to(p_copy.dtype)
        p_copy = p_copy * has_src

        p_vocab = F.softmax(vocab_logits, dim=-1)
        p_copy_vocab = torch.zeros_like(p_vocab)
        scatter_idx = copy_token_ids.unsqueeze(1).expand(bsz, seq_len, seq_len)
        p_copy_vocab.scatter_add_(2, scatter_idx, attn)

        p_final = (1.0 - p_copy) * p_vocab + p_copy * p_copy_vocab
        return torch.log(p_final.clamp_min(1e-8))


def extract_prompt_entities(prompt: str) -> list[tuple[str, str]]:
    """Retorna [(text, kind)] na mesma ordem determinística do `isa_bank.rs`."""
    order: list[str] = []
    kinds: dict[str, str] = {}

    def is_citable(k: str) -> bool:
        return k in ("ReadSpan", "VerifiedFact", "CandidateFact")

    def register(raw: str, kind: str) -> None:
        s = raw.strip()
        if not s:
            return
        if s in kinds:
            if is_citable(kind) or not is_citable(kinds[s]):
                kinds[s] = kind
        else:
            kinds[s] = kind
            order.append(s)

    def register_fact(f: dict, id_kind: str) -> None:
        if isinstance(f.get("id"), str):
            register(f["id"], id_kind)
        for key in ("subject", "relation", "object"):
            if isinstance(f.get(key), str):
                register(f[key], "PromptEntity")

    for line in prompt.splitlines():
        t = line.strip()
        cand_body = t[7:].strip() if t.startswith("SYSTEM:") else t
        if cand_body.startswith("CANDIDATES:"):
            try:
                arr = json.loads(cand_body[11:].strip())
            except json.JSONDecodeError:
                continue
            if isinstance(arr, list):
                for f in arr:
                    if isinstance(f, dict):
                        register_fact(f, "CandidateFact")
        elif t.startswith("TOOL:"):
            try:
                obj = json.loads(t[5:].strip())
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                if isinstance(obj.get("fact"), dict):
                    register_fact(obj["fact"], "VerifiedFact")
                if isinstance(obj.get("facts"), list):
                    for f in obj["facts"]:
                        if isinstance(f, dict):
                            register_fact(f, "VerifiedFact")
                for sp in obj.get("spans", []) if isinstance(obj.get("spans"), list) else []:
                    if isinstance(sp, dict) and isinstance(sp.get("id"), str):
                        register(sp["id"], "ReadSpan")
                for m in obj.get("matches", []) if isinstance(obj.get("matches"), list) else []:
                    if isinstance(m, dict) and isinstance(m.get("id"), str):
                        register(m["id"], "DocumentToc")
        elif t.startswith("USER:"):
            for w in t[5:].split():
                w = w.strip(".,?!;:()\"'")
                if len(w) >= 2:
                    register(w, "PromptEntity")
        elif t.startswith("[") and "]" in t:
            sid = t[1 : t.index("]")]
            if sid and not any(ch.isspace() for ch in sid):
                register(sid, "DocumentToc")

    return [(s, kinds[s]) for s in order]


def find_entity_token_span(
    haystack_bytes: bytes,
    needle: str,
    tok_byte_ends: list[int],
) -> tuple[int, int] | None:
    nb = needle.encode("utf-8")
    b0 = haystack_bytes.rfind(nb)
    if b0 < 0:
        return None
    b1 = b0 + len(nb)
    t0: int | None = None
    t1 = 0
    prev = 0
    for idx, end in enumerate(tok_byte_ends):
        if end > b0 and prev < b1:
            if t0 is None:
                t0 = idx
            t1 = idx + 1
        prev = end
    return None if t0 is None else (t0, max(t1, t0 + 1))


def compute_isa_sample_loss(
    model: NeuroSymRwkv7,
    tok: RwkvWorldTokenizer,
    formatted_prompt: str,
    action_dict: dict,
    device: torch.device,
) -> torch.Tensor:
    assert model.isa is not None
    isa = model.isa
    cfg = model.cfg

    p_ids = tok.encode(formatted_prompt)
    prompt_seq = torch.tensor([[cfg.bos_token_id, *p_ids]], dtype=torch.long, device=device)

    # Prefill único do prompt sem gradiente, devolvendo o estado O(1) para continuar só na ISA
    with torch.no_grad():
        h_all, prompt_states = model.forward_hidden_embs(model.emb(prompt_seq))
        h_prompt = h_all[0, 1:, :]          # [P, d] para pooling e_s
        h_last_prompt = h_all[0, -1:, :]    # [1, d] estado que prevê <OP>
        prompt_states = [
            (px.detach(), mat.detach(), cx.detach()) for (px, mat, cx) in prompt_states
        ]

    prompt_bytes = formatted_prompt.encode("utf-8")
    byte_ends: list[int] = []
    cur = 0
    for tid in p_ids:
        cur += len(tok.idx2bytes[tid])
        byte_ends.append(cur)

    # 1. Monta banco neural de entidades (gradiente flui em ptr_k, ptr_in, ptr_q)
    slot_texts: list[str] = []
    slot_kinds: list[str] = []
    pooled_list: list[torch.Tensor] = []
    text_to_slot: dict[str, int] = {}

    for ent_text, kind in extract_prompt_entities(formatted_prompt):
        span = find_entity_token_span(prompt_bytes, ent_text, byte_ends)
        if span is not None:
            t0, t1 = span
            text_to_slot[ent_text] = len(slot_texts)
            slot_texts.append(ent_text)
            slot_kinds.append(kind)
            pooled_list.append(h_prompt[t0:t1].mean(dim=0).detach())

    if pooled_list:
        e_mat = torch.stack(pooled_list, dim=0)
        k_mat = isa.ptr_k(e_mat)
    else:
        k_mat = torch.zeros((0, cfg.isa_ptr_dim), device=device, dtype=h_prompt.dtype)

    # 2. Codifica a ação alvo em lista de passos (tipo_passo, target_info, feedback_emb)
    op_name = action_dict["action"]
    op_idx = ISA_OP_TO_IDX[op_name]
    steps: list[tuple[str, object, torch.Tensor]] = []

    def add_lit_steps(text: str) -> None:
        for tid in tok.encode(text):
            steps.append(("lit_tok", tid, model.emb.weight[tid]))
        steps.append(("lit_tok", cfg.vocab_size, isa.ctl_emb.weight[CTL_LIT_CLOSE]))

    def add_ref_step(value: str, allowed_filter: str, allow_end: bool, strict: bool) -> None:
        if value in text_to_slot:
            sid = text_to_slot[value]
            fb = isa.ptr_tag + isa.ptr_in(k_mat[sid])
            steps.append(("ref", (sid, allowed_filter, allow_end, strict), fb))
        else:
            steps.append(
                ("ref", ("lit_open", allowed_filter, allow_end, False), isa.ctl_emb.weight[CTL_LIT_OPEN])
            )
            add_lit_steps(value)

    steps.append(("op", op_idx, isa.op_emb.weight[op_idx]))
    if op_name == "query_fact":
        add_ref_step(action_dict["entity"], "any", False, False)
        add_ref_step(action_dict["relation"], "any", False, False)
    elif op_name in ("search_facts", "search_context"):
        add_ref_step(action_dict["query"], "any", False, False)
        lim = min(int(action_dict["limit"]), cfg.isa_num_vocab - 1)
        steps.append(("num", lim, isa.num_emb.weight[lim]))
    elif op_name == "traverse_graph":
        add_ref_step(action_dict["start_entity"], "any", False, False)
        hops = min(int(action_dict["max_hops"]), cfg.isa_num_vocab - 1)
        steps.append(("num", hops, isa.num_emb.weight[hops]))
    elif op_name == "eval_math":
        add_lit_steps(action_dict["expression"])
    elif op_name == "check_logic":
        add_lit_steps(action_dict["smtlib2_script"])
    elif op_name == "run_wasm_check":
        add_lit_steps(action_dict["wasm_wat_source"])
        add_lit_steps(action_dict["expected_stdout"])
    elif op_name == "read_spans":
        for sid in action_dict["span_ids"]:
            add_ref_step(sid, "doc_span", True, True)
        steps.append(("ref", ("end", "doc_span", True, True), isa.ctl_emb.weight[CTL_END]))
    elif op_name == "propose_answer":
        add_lit_steps(action_dict["text"])
        for fid in action_dict["cited_facts"]:
            add_ref_step(fid, "citable", True, True)
        steps.append(("ref", ("end", "citable", True, True), isa.ctl_emb.weight[CTL_END]))

    # 3. Continua direto do estado O(1) do prompt rodando SOMENTE os passos da ISA
    if len(steps) > 1:
        isa_in = torch.stack([s[2] for s in steps[:-1]], dim=0).unsqueeze(0).to(h_last_prompt.dtype)
        h_isa, _ = model.forward_hidden_embs(isa_in, states=prompt_states)
        h_steps = torch.cat([h_last_prompt, h_isa[0]], dim=0)
    else:
        h_steps = h_last_prompt

    losses: list[torch.Tensor] = []
    inv_sqrt = 1.0 / math.sqrt(cfg.isa_ptr_dim)

    for step_idx, (kind, target, _) in enumerate(steps):
        h_t = h_steps[step_idx]
        if kind == "op":
            logits = isa.op_head(h_t)
            losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([target], device=device)))
        elif kind == "num":
            logits = isa.num_head(h_t)
            losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([target], device=device)))
        elif kind == "lit_tok":
            v_logits = model.head(h_t)
            close_logit = isa.ctl_head(h_t)[CTL_LIT_CLOSE : CTL_LIT_CLOSE + 1]
            logits = torch.cat([v_logits, close_logit], dim=0)
            losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([target], device=device)))
        elif kind == "ref":
            tgt, filt, allow_end, strict = target  # type: ignore[misc]
            q_ptr = isa.ptr_q(h_t)
            cand_logits: list[torch.Tensor] = []
            target_pos = -1

            for s_i, s_kind in enumerate(slot_kinds):
                allowed = (
                    filt == "any"
                    or (filt == "citable" and s_kind in ("ReadSpan", "VerifiedFact", "CandidateFact"))
                    or (filt == "doc_span" and s_kind in ("DocumentToc", "ReadSpan"))
                )
                if not allowed:
                    continue
                if tgt == s_i:
                    target_pos = len(cand_logits)
                cand_logits.append((q_ptr * k_mat[s_i]).sum() * inv_sqrt)

            if allow_end:
                if tgt == "end":
                    target_pos = len(cand_logits)
                cand_logits.append(isa.ctl_head(h_t)[CTL_END])

            if not strict:
                if tgt == "lit_open":
                    target_pos = len(cand_logits)
                cand_logits.append(isa.ctl_head(h_t)[CTL_LIT_OPEN])

            if target_pos >= 0 and len(cand_logits) > 1:
                logits = torch.stack(cand_logits, dim=0)
                losses.append(
                    F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_pos], device=device))
                )

    return torch.stack(losses).mean()


def load_official_blinkdl_pth(
    pth_path: Path,
    copy_dim: int,
    copy_max_len: int,
    isa_ptr_dim: int = 0,
    isa_num_vocab: int = 64,
) -> tuple[NeuroSymRwkv7, Rwkv7Config]:
    """Lê qualquer `.pth` oficial do BlinkDL RWKV-7 (0.1B, 0.4B, 1.5B...) e mapeia 1:1."""
    raw: dict[str, torch.Tensor] = torch.load(pth_path, map_location="cpu", weights_only=True)

    vocab_size, d_model = raw["emb.weight"].shape
    n_layers = sum(1 for k in raw if re.fullmatch(r"blocks\.\d+\.ln1\.weight", k))
    head_dim = raw["blocks.0.att.r_k"].shape[-1]
    ffn_dim = raw["blocks.0.ffn.key.weight"].shape[0]
    decay_lora_dim = raw["blocks.0.att.w1"].shape[1]
    a_lora_dim = raw["blocks.0.att.a1"].shape[1]
    gate_lora_dim = raw["blocks.0.att.g1"].shape[1]
    v_lora_dim = raw["blocks.1.att.v1"].shape[1] if "blocks.1.att.v1" in raw else 32

    cfg = Rwkv7Config(
        vocab_size=int(vocab_size),
        d_model=int(d_model),
        n_layers=int(n_layers),
        head_dim=int(head_dim),
        ffn_dim=int(ffn_dim),
        decay_lora_dim=int(decay_lora_dim),
        a_lora_dim=int(a_lora_dim),
        v_lora_dim=int(v_lora_dim),
        gate_lora_dim=int(gate_lora_dim),
        copy_dim=copy_dim,
        copy_max_len=copy_max_len,
        isa_ptr_dim=isa_ptr_dim,
        isa_num_vocab=isa_num_vocab,
    )

    model = NeuroSymRwkv7(cfg)
    with torch.no_grad():
        model.emb.weight.copy_(raw["emb.weight"].float())
        model.ln_in.weight.copy_(raw["blocks.0.ln0.weight"].float())
        model.ln_in.bias.copy_(raw["blocks.0.ln0.bias"].float())
        model.ln_out.weight.copy_(raw["ln_out.weight"].float())
        model.ln_out.bias.copy_(raw["ln_out.bias"].float())
        model.head.weight.copy_(raw["head.weight"].float())

        for i in range(n_layers):
            bp = f"blocks.{i}"
            blk = model.blocks[i]
            blk.ln1.weight.copy_(raw[f"{bp}.ln1.weight"].float())
            blk.ln1.bias.copy_(raw[f"{bp}.ln1.bias"].float())
            blk.ln2.weight.copy_(raw[f"{bp}.ln2.weight"].float())
            blk.ln2.bias.copy_(raw[f"{bp}.ln2.bias"].float())

            tm = blk.tm
            for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "w0", "a0", "k_k", "k_a"):
                getattr(tm, name).copy_(raw[f"{bp}.att.{name}"].float().reshape(d_model))

            tm.r_k.copy_(raw[f"{bp}.att.r_k"].float().reshape(d_model))
            tm.ln_x.weight.copy_(raw[f"{bp}.att.ln_x.weight"].float().reshape(d_model))
            tm.ln_x.bias.copy_(raw[f"{bp}.att.ln_x.bias"].float().reshape(d_model))

            tm.receptance.weight.copy_(raw[f"{bp}.att.receptance.weight"].float())
            tm.key.weight.copy_(raw[f"{bp}.att.key.weight"].float())
            tm.value.weight.copy_(raw[f"{bp}.att.value.weight"].float())
            tm.output.weight.copy_(raw[f"{bp}.att.output.weight"].float())

            tm.w1.weight.copy_(raw[f"{bp}.att.w1"].float().T)
            tm.w2.weight.copy_(raw[f"{bp}.att.w2"].float().T)
            tm.a1.weight.copy_(raw[f"{bp}.att.a1"].float().T)
            tm.a2.weight.copy_(raw[f"{bp}.att.a2"].float().T)
            tm.g1.weight.copy_(raw[f"{bp}.att.g1"].float().T)
            tm.g2.weight.copy_(raw[f"{bp}.att.g2"].float().T)

            if i > 0 and tm.v0 is not None and tm.v1 is not None and tm.v2 is not None:
                tm.v0.copy_(raw[f"{bp}.att.v0"].float().reshape(d_model))
                tm.v1.weight.copy_(raw[f"{bp}.att.v1"].float().T)
                tm.v2.weight.copy_(raw[f"{bp}.att.v2"].float().T)

            cm = blk.cm
            cm.x_k.copy_(raw[f"{bp}.ffn.x_k"].float().reshape(d_model))
            cm.key.weight.copy_(raw[f"{bp}.ffn.key.weight"].float())
            cm.value.weight.copy_(raw[f"{bp}.ffn.value.weight"].float())

    return model, cfg


def format_neurosym_prompt(raw_prompt: str) -> str:
    """Alinha com o estilo de chat do RWKV-7 G1 (sem espaço final após `Assistant:`)."""
    cleaned = raw_prompt.strip().replace("\r\n", "\n")
    return f"{cleaned}\nAssistant:"


def augment_copy_identifiers(prompt: str, action_json: str, rng: random.Random) -> tuple[str, str]:
    new_prompt, new_action = prompt, action_json
    for fid in set(RE_SYNTH_FACT_ID.findall(new_action)):
        if fid in new_prompt:
            rand_id = (
                f"{rng.choice(AUG_FACT_PREFIXES)}-{rng.choice(AUG_FACT_TAGS)}-{rng.randint(1000, 9999)}"
            )
            new_prompt = new_prompt.replace(fid, rand_id)
            new_action = new_action.replace(fid, rand_id)
    for ent in set(RE_SYNTH_ENTITY.findall(new_action)):
        if ent in new_prompt:
            rand_ent = (
                f"{rng.choice(AUG_ENTITY_PREFIXES)}_{rng.choice(AUG_ENTITY_SUFFIXES)}_{rng.randint(10, 99)}"
            )
            new_prompt = new_prompt.replace(ent, rand_ent)
            new_action = new_action.replace(ent, rand_ent)
    return new_prompt, new_action


def make_batch(
    batch_items: list[tuple[list[int], int]],
    pad_id: int,
    copy_max_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq, _ in batch_items)
    slots = max_len - 1
    bsz = len(batch_items)

    inputs = torch.full((bsz, slots), pad_id, dtype=torch.long, device=device)
    targets = torch.full((bsz, slots), -100, dtype=torch.long, device=device)
    src_mask = torch.zeros((bsz, slots), dtype=torch.bool, device=device)
    copy_ids = torch.zeros((bsz, slots), dtype=torch.long, device=device)

    for row, (seq, prompt_end) in enumerate(batch_items):
        n = len(seq)
        inputs[row, : n - 1] = torch.tensor(seq[:-1], dtype=torch.long, device=device)
        targets[row, prompt_end - 1 : n - 1] = torch.tensor(
            seq[prompt_end:], dtype=torch.long, device=device
        )

        src_end = max(0, prompt_end - 1)
        src_start = max(0, src_end - copy_max_len) if copy_max_len > 0 else src_end
        if src_end > src_start:
            src_mask[row, src_start:src_end] = True
            copy_ids[row, src_start:src_end] = torch.tensor(
                seq[src_start + 1 : src_end + 1], dtype=torch.long, device=device
            )

    return inputs, targets, src_mask, copy_ids


def export_state_dict(
    model: nn.Module,
    quantize_int8: bool,
    quantize_emb_int8: bool,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for name, param in model.state_dict().items():
        cpu_f32 = param.detach().to(device="cpu", dtype=torch.float32).contiguous()

        should_q = (
            cpu_f32.ndim == 2
            and not name.startswith("isa.")
            and (
                (quantize_emb_int8 and name == "emb.weight")
                or (
                    quantize_int8
                    and (name == "head.weight" or name.endswith(LINEAR_WEIGHT_SUFFIXES))
                )
            )
        )
        if should_q:
            max_abs = cpu_f32.abs().amax(dim=1)
            scales = max_abs / 127.0
            safe_scales = torch.where(scales == 0.0, torch.ones_like(scales), scales)
            quantized = (
                torch.round(cpu_f32 / safe_scales.unsqueeze(1))
                .clamp(-127, 127)
                .to(torch.int8)
                .contiguous()
            )
            tensors[f"{name}.q"] = quantized
            tensors[f"{name}.scale"] = scales.to(torch.float32).contiguous()
        else:
            tensors[name] = cpu_f32
    return tensors


def run_pipeline(args: argparse.Namespace) -> None:
    print(f"[RWKV-7 G1] Carregando checkpoint oficial: {args.pth}", flush=True)
    isa_ptr_dim = args.isa_ptr_dim if args.isa else 0
    copy_dim = 0 if (args.convert_only or args.isa) else args.copy_dim

    model, cfg = load_official_blinkdl_pth(
        args.pth,
        copy_dim=copy_dim,
        copy_max_len=args.copy_max_len,
        isa_ptr_dim=isa_ptr_dim,
        isa_num_vocab=args.isa_num_vocab,
    )
    tok = RwkvWorldTokenizer(args.vocab_txt, vocab_size=cfg.vocab_size)
    if model.isa is not None:
        model.isa.semantic_init(model.emb.weight, tok)

    print(
        f"[RWKV-7 G1] Arquitetura detectada: L={cfg.n_layers}, d={cfg.d_model}, "
        f"head_dim={cfg.head_dim}, ffn={cfg.ffn_dim}, V={cfg.vocab_size}, "
        f"isa_ptr_dim={cfg.isa_ptr_dim}, copy_dim={cfg.copy_dim}",
        flush=True,
    )

    if not args.convert_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # train_last_layers:
        #   < 0  -> treina todas as camadas
        #   == 0 -> congela os 12 blocos (treina apenas cabeça ISA/Copy + ln_out)
        #   1..L -> descongela apenas as últimas N camadas
        if 0 <= args.train_last_layers < cfg.n_layers:
            for param in model.parameters():
                param.requires_grad = False
            if args.train_last_layers > 0:
                start_unfreeze = cfg.n_layers - args.train_last_layers
                for i in range(start_unfreeze, cfg.n_layers):
                    for param in model.blocks[i].parameters():
                        param.requires_grad = True
            unfreeze_mods = [model.ln_out, model.copy_q, model.copy_k, model.copy_gate, model.isa]
            if not args.freeze_head:
                unfreeze_mods.append(model.head)
            for mod in unfreeze_mods:
                if mod is not None:
                    for param in mod.parameters():
                        param.requires_grad = True
        elif args.freeze_head:
            for param in model.head.parameters():
                param.requires_grad = False

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        trainable = sum(p.numel() for p in trainable_params)
        total = sum(p.numel() for p in model.parameters())
        mode = "ISA" if args.isa else "JSON"
        print(
            f"[RWKV-7 SFT {mode}] device={device} | fla_triton={HAS_FLA_RWKV7 and device.type == 'cuda'} | "
            f"trainable={trainable:,}/{total:,} ({100.0 * trainable / total:.2f}%)",
            flush=True,
        )

        raw_rows: list[tuple[str, dict]] = []
        with args.dataset.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                raw_rows.append((format_neurosym_prompt(row["prompt"]), row["action"]))

        rng = random.Random(42)
        if args.max_samples > 0 and len(raw_rows) > args.max_samples:
            rng.shuffle(raw_rows)
            raw_rows = raw_rows[: args.max_samples]
            print(
                f"[RWKV-7 SFT {mode}] Usando subconjunto de {len(raw_rows)} amostras (--max-samples)",
                flush=True,
            )

        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

        if args.isa:
            for epoch in range(1, args.epochs + 1):
                model.train()
                order = list(range(len(raw_rows)))
                rng.shuffle(order)

                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                steps_done = 0
                accum = 0
                t_ep = time.perf_counter()

                for step_i, idx in enumerate(order, start=1):
                    p, action_dict = raw_rows[idx]
                    if rng.random() < 0.65:
                        a_json = json.dumps(action_dict, ensure_ascii=False, separators=(",", ":"))
                        p, a_json = augment_copy_identifiers(p, a_json, rng)
                        action_dict = json.loads(a_json)

                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=(device.type == "cuda"),
                    ):
                        sample_loss = compute_isa_sample_loss(model, tok, p, action_dict, device)

                    (sample_loss / args.batch_size).backward()
                    loss_sum += sample_loss.item()
                    steps_done += 1
                    accum += 1

                    if accum >= args.batch_size or step_i == len(order):
                        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        accum = 0

                    if step_i % 10 == 0 or step_i == len(order):
                        elapsed = time.perf_counter() - t_ep
                        sec_per_item = elapsed / step_i
                        eta_s = sec_per_item * (len(order) - step_i)
                        print(
                            f"  epoch {epoch:02d}/{args.epochs} | {step_i:4d}/{len(order)} | "
                            f"isa_loss={loss_sum / max(1, steps_done):.4f} | "
                            f"{sec_per_item:.2f}s/amostra | eta={eta_s:.0f}s",
                            flush=True,
                        )

                scheduler.step()
        else:
            raw_samples = [
                (p, json.dumps(a, ensure_ascii=False, separators=(",", ":")))
                for p, a in raw_rows
            ]
            for epoch in range(1, args.epochs + 1):
                model.train()
                encoded: list[tuple[list[int], int]] = []
                for p, a in raw_samples:
                    if cfg.copy_dim > 0 and rng.random() < 0.65:
                        p, a = augment_copy_identifiers(p, a, rng)
                    p_ids = tok.encode(p)
                    a_ids = tok.encode(" " + a)
                    seq = [cfg.bos_token_id, *p_ids, *a_ids, cfg.eos_token_id]
                    encoded.append((seq, 1 + len(p_ids)))

                encoded.sort(key=lambda item: len(item[0]) + rng.randint(-6, 6))
                batches = [
                    encoded[i : i + args.batch_size]
                    for i in range(0, len(encoded), args.batch_size)
                ]
                rng.shuffle(batches)

                loss_sum = 0.0
                for chunk in batches:
                    x_in, y_tgt, src_mask, copy_ids = make_batch(
                        chunk, cfg.eos_token_id, cfg.copy_max_len, device
                    )
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=(device.type == "cuda"),
                    ):
                        log_probs = model(x_in, src_mask, copy_ids)
                        loss = F.nll_loss(
                            log_probs.reshape(-1, cfg.vocab_size),
                            y_tgt.reshape(-1),
                            ignore_index=-100,
                        )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    loss_sum += loss.item()

                scheduler.step()
                print(
                    f"  epoch {epoch:02d}/{args.epochs} | train_loss = {loss_sum / max(1, len(batches)):.5f}",
                    flush=True,
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    tok.export_binary(args.out_dir / cfg.vocab_file)

    tensors = export_state_dict(
        model,
        quantize_int8=args.quantize_int8,
        quantize_emb_int8=args.quantize_emb_int8,
    )
    out_weights = args.out_dir / "model.safetensors"
    save_file(tensors, str(out_weights))
    size_mib = out_weights.stat().st_size / (1024.0 * 1024.0)
    print(
        f"[RWKV-7 G1] Exportado em {args.out_dir} | model.safetensors = {size_mib:.2f} MiB | "
        f"isa={args.isa} | quantize_int8={args.quantize_int8} | quantize_emb_int8={args.quantize_emb_int8}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converte e faz SFT do RWKV-7 G1 0.1B para o NeuroSym Rust"
    )
    parser.add_argument("--pth", type=Path, required=True, help="Caminho do .pth oficial do BlinkDL/rwkv7-g1")
    parser.add_argument("--vocab-txt", type=Path, required=True, help="Caminho do rwkv_vocab_v20230424.txt")
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/verified_dataset.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/rwkv7_g1_01b_ns"))
    parser.add_argument("--convert-only", action="store_true", help="Apenas converte e quantiza sem rodar SFT")
    parser.add_argument("--isa", action="store_true", help="Ativa a Dual-Head ISA + Entity Pointer")
    parser.add_argument("--isa-ptr-dim", type=int, default=64)
    parser.add_argument("--isa-num-vocab", type=int, default=64)
    parser.add_argument("--freeze-head", action="store_true", help="Congela head.weight (50.3M params)")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = usa o dataset inteiro")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--train-last-layers",
        type=int,
        default=3,
        help="0 = congela os 12 blocos (só cabeças); -1 = treina todas as camadas",
    )
    parser.add_argument("--copy-dim", type=int, default=64)
    parser.add_argument("--copy-max-len", type=int, default=512)
    parser.add_argument("--quantize-int8", action="store_true")
    parser.add_argument("--quantize-emb-int8", action="store_true")
    run_pipeline(parser.parse_args())
