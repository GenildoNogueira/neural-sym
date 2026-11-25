//! Codec `Action <-> ISA`: representação compacta de ações para o modelo.
//!
//! Ideia: o modelo não emite JSON; emite uma sequência curta de símbolos
//! (`Op`, `Ptr`, `Lit`, `Num`, `End`). O Rust converte de/para o JSON canônico.
//!
//! Propriedades garantidas (ver testes):
//! 1. `decode(encode(a, bank), bank) == a` para qualquer ação e qualquer banco
//!    (o fallback `Lit` torna o codec sem perdas mesmo sem ponteiros).
//! 2. Passar por texto (`to_text` / `from_text`) também é sem perdas.
//! 3. Ter um banco nunca aumenta o custo.
//! 4. `expect()` (gramática/máscara) nunca proíbe a continuação verdadeira.
//! 5. Nenhuma entrada arbitrária causa panic.
//!
//! Este módulo só depende de `std`. `Action` espelha `ns_core::action::LlmAction`
//! (ver `isa_adapter.rs` para as conversões).

use std::collections::HashMap;
use std::fmt;

pub type Slot = u16;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    QueryFact {
        entity: String,
        relation: String,
    },
    SearchFacts {
        query: String,
        limit: u8,
    },
    TraverseGraph {
        start_entity: String,
        max_hops: u8,
    },
    EvalMath {
        expression: String,
    },
    ProposeAnswer {
        text: String,
        cited_facts: Vec<String>,
    },
    SearchContext {
        query: String,
        limit: u8,
    },
    ReadSpans {
        span_ids: Vec<String>,
    },
    CheckLogic {
        smtlib2_script: String,
    },
    RunWasmCheck {
        wasm_wat_source: String,
        expected_stdout: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    QueryFact,
    SearchFacts,
    TraverseGraph,
    EvalMath,
    ProposeAnswer,
    SearchContext,
    ReadSpans,
    CheckLogic,
    RunWasmCheck,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Ref,
    Lit,
    Num,
    RefList,
}

impl Op {
    pub const ALL: [Op; 9] = [
        Op::QueryFact,
        Op::SearchFacts,
        Op::TraverseGraph,
        Op::EvalMath,
        Op::ProposeAnswer,
        Op::SearchContext,
        Op::ReadSpans,
        Op::CheckLogic,
        Op::RunWasmCheck,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Op::QueryFact => "QUERY",
            Op::SearchFacts => "SEARCH",
            Op::TraverseGraph => "TRAVERSE",
            Op::EvalMath => "EVAL",
            Op::ProposeAnswer => "ANSWER",
            Op::SearchContext => "SEARCH_CTX",
            Op::ReadSpans => "READ",
            Op::CheckLogic => "LOGIC",
            Op::RunWasmCheck => "WASM",
        }
    }

    pub fn from_name(s: &str) -> Option<Op> {
        Op::ALL.iter().copied().find(|op| op.name() == s)
    }

    /// Forma dos argumentos de cada operação: única fonte de verdade para
    /// `decode_one` e `expect`.
    fn shape(self) -> &'static [Kind] {
        match self {
            Op::QueryFact => &[Kind::Ref, Kind::Ref],
            Op::SearchFacts | Op::TraverseGraph | Op::SearchContext => &[Kind::Ref, Kind::Num],
            Op::EvalMath | Op::CheckLogic => &[Kind::Lit],
            Op::RunWasmCheck => &[Kind::Lit, Kind::Lit],
            Op::ProposeAnswer => &[Kind::Lit, Kind::RefList],
            Op::ReadSpans => &[Kind::RefList],
        }
    }
}

// Símbolos

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Sym {
    Op(Op),
    /// Ponteiro para uma entrada do [`Bank`] (1 passo de decode).
    Ptr(Slot),
    /// Texto literal (fallback e campos de texto livre).
    Lit(String),
    Num(u8),
    End,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IsaError {
    Incomplete,
    Unexpected { pos: usize, want: &'static str },
    BadSlot(Slot),
    Text(&'static str),
    Internal,
}

impl fmt::Display for IsaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IsaError::Incomplete => write!(f, "sequência incompleta"),
            IsaError::Unexpected { pos, want } => {
                write!(f, "símbolo inesperado em {pos}: esperado {want}")
            }
            IsaError::BadSlot(k) => write!(f, "slot {k} fora do banco"),
            IsaError::Text(m) => write!(f, "texto ISA inválido: {m}"),
            IsaError::Internal => write!(f, "erro interno do codec"),
        }
    }
}

impl std::error::Error for IsaError {}

// Banco

/// Tabela de entidades endereçáveis (ids de fatos, entidades, cláusulas...).
/// Na prática é montada a partir do prompt e dos resultados de TOOL anteriores.
#[derive(Clone, Debug, Default)]
pub struct Bank {
    items: Vec<String>,
    index: HashMap<String, Slot>,
}

impl Bank {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insere (ou reaproveita) e devolve o slot. `None` se o banco encheu.
    pub fn intern(&mut self, s: &str) -> Option<Slot> {
        if let Some(&k) = self.index.get(s) {
            return Some(k);
        }
        let k = Slot::try_from(self.items.len()).ok()?;
        self.items.push(s.to_owned());
        self.index.insert(s.to_owned(), k);
        Some(k)
    }

    pub fn slot_of(&self, s: &str) -> Option<Slot> {
        self.index.get(s).copied()
    }

    pub fn get(&self, k: Slot) -> Option<&str> {
        self.items.get(k as usize).map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

// Encode

fn push_ref(out: &mut Vec<Sym>, s: &str, bank: &Bank) {
    match bank.slot_of(s) {
        Some(k) => out.push(Sym::Ptr(k)),
        None => out.push(Sym::Lit(s.to_owned())),
    }
}

pub fn encode(action: &Action, bank: &Bank) -> Vec<Sym> {
    let mut out = Vec::with_capacity(6);
    match action {
        Action::QueryFact { entity, relation } => {
            out.push(Sym::Op(Op::QueryFact));
            push_ref(&mut out, entity, bank);
            push_ref(&mut out, relation, bank);
        }
        Action::SearchFacts { query, limit } => {
            out.push(Sym::Op(Op::SearchFacts));
            push_ref(&mut out, query, bank);
            out.push(Sym::Num(*limit));
        }
        Action::TraverseGraph {
            start_entity,
            max_hops,
        } => {
            out.push(Sym::Op(Op::TraverseGraph));
            push_ref(&mut out, start_entity, bank);
            out.push(Sym::Num(*max_hops));
        }
        Action::EvalMath { expression } => {
            out.push(Sym::Op(Op::EvalMath));
            out.push(Sym::Lit(expression.clone()));
        }
        Action::ProposeAnswer { text, cited_facts } => {
            out.push(Sym::Op(Op::ProposeAnswer));
            out.push(Sym::Lit(text.clone()));
            for id in cited_facts {
                push_ref(&mut out, id, bank);
            }
            out.push(Sym::End);
        }
        Action::SearchContext { query, limit } => {
            out.push(Sym::Op(Op::SearchContext));
            push_ref(&mut out, query, bank);
            out.push(Sym::Num(*limit));
        }
        Action::ReadSpans { span_ids } => {
            out.push(Sym::Op(Op::ReadSpans));
            for id in span_ids {
                push_ref(&mut out, id, bank);
            }
            out.push(Sym::End);
        }
        Action::CheckLogic { smtlib2_script } => {
            out.push(Sym::Op(Op::CheckLogic));
            out.push(Sym::Lit(smtlib2_script.clone()));
        }
        Action::RunWasmCheck {
            wasm_wat_source,
            expected_stdout,
        } => {
            out.push(Sym::Op(Op::RunWasmCheck));
            out.push(Sym::Lit(wasm_wat_source.clone()));
            out.push(Sym::Lit(expected_stdout.clone()));
        }
    }
    out
}

// Decode

enum Arg {
    Str(String),
    Num(u8),
    List(Vec<String>),
}

struct Args(std::vec::IntoIter<Arg>);

impl Args {
    fn str(&mut self) -> Result<String, IsaError> {
        match self.0.next() {
            Some(Arg::Str(s)) => Ok(s),
            _ => Err(IsaError::Internal),
        }
    }
    fn num(&mut self) -> Result<u8, IsaError> {
        match self.0.next() {
            Some(Arg::Num(n)) => Ok(n),
            _ => Err(IsaError::Internal),
        }
    }
    fn list(&mut self) -> Result<Vec<String>, IsaError> {
        match self.0.next() {
            Some(Arg::List(v)) => Ok(v),
            _ => Err(IsaError::Internal),
        }
    }
}

fn read_ref(syms: &[Sym], pos: &mut usize, bank: &Bank) -> Result<String, IsaError> {
    match syms.get(*pos) {
        Some(Sym::Ptr(k)) => {
            let s = bank.get(*k).ok_or(IsaError::BadSlot(*k))?.to_owned();
            *pos += 1;
            Ok(s)
        }
        Some(Sym::Lit(s)) => {
            *pos += 1;
            Ok(s.clone())
        }
        Some(_) => Err(IsaError::Unexpected {
            pos: *pos,
            want: "ptr ou literal",
        }),
        None => Err(IsaError::Incomplete),
    }
}

/// Decodifica uma ação do início de `syms`. Devolve a ação e quantos símbolos consumiu.
pub fn decode_one(syms: &[Sym], bank: &Bank) -> Result<(Action, usize), IsaError> {
    let mut pos = 0usize;
    let op = match syms.get(pos) {
        Some(Sym::Op(op)) => *op,
        Some(_) => return Err(IsaError::Unexpected { pos, want: "op" }),
        None => return Err(IsaError::Incomplete),
    };
    pos += 1;

    let mut args: Vec<Arg> = Vec::with_capacity(2);
    for kind in op.shape() {
        match kind {
            Kind::Ref => args.push(Arg::Str(read_ref(syms, &mut pos, bank)?)),
            Kind::Lit => match syms.get(pos) {
                Some(Sym::Lit(s)) => {
                    args.push(Arg::Str(s.clone()));
                    pos += 1;
                }
                Some(_) => {
                    return Err(IsaError::Unexpected {
                        pos,
                        want: "literal",
                    });
                }
                None => return Err(IsaError::Incomplete),
            },
            Kind::Num => match syms.get(pos) {
                Some(Sym::Num(n)) => {
                    args.push(Arg::Num(*n));
                    pos += 1;
                }
                Some(_) => {
                    return Err(IsaError::Unexpected {
                        pos,
                        want: "número",
                    });
                }
                None => return Err(IsaError::Incomplete),
            },
            Kind::RefList => {
                let mut v = Vec::new();
                loop {
                    match syms.get(pos) {
                        Some(Sym::End) => {
                            pos += 1;
                            break;
                        }
                        Some(_) => v.push(read_ref(syms, &mut pos, bank)?),
                        None => return Err(IsaError::Incomplete),
                    }
                }
                args.push(Arg::List(v));
            }
        }
    }

    let mut a = Args(args.into_iter());
    let action = match op {
        Op::QueryFact => Action::QueryFact {
            entity: a.str()?,
            relation: a.str()?,
        },
        Op::SearchFacts => Action::SearchFacts {
            query: a.str()?,
            limit: a.num()?,
        },
        Op::TraverseGraph => Action::TraverseGraph {
            start_entity: a.str()?,
            max_hops: a.num()?,
        },
        Op::EvalMath => Action::EvalMath {
            expression: a.str()?,
        },
        Op::ProposeAnswer => Action::ProposeAnswer {
            text: a.str()?,
            cited_facts: a.list()?,
        },
        Op::SearchContext => Action::SearchContext {
            query: a.str()?,
            limit: a.num()?,
        },
        Op::ReadSpans => Action::ReadSpans {
            span_ids: a.list()?,
        },
        Op::CheckLogic => Action::CheckLogic {
            smtlib2_script: a.str()?,
        },
        Op::RunWasmCheck => Action::RunWasmCheck {
            wasm_wat_source: a.str()?,
            expected_stdout: a.str()?,
        },
    };
    Ok((action, pos))
}

/// Decodifica uma sequência inteira de ações (banco fixo).
pub fn decode_stream(syms: &[Sym], bank: &Bank) -> Result<Vec<Action>, IsaError> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < syms.len() {
        let (a, used) = decode_one(&syms[pos..], bank)?;
        out.push(a);
        pos += used;
    }
    Ok(out)
}

// Gramática (máscara)

/// O que a gramática permite como PRÓXIMO símbolo. Substitui o JSON Schema do llguidance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expect {
    Op,
    Ref,
    Lit,
    Num,
    RefOrEnd,
}

impl Expect {
    /// `allow_lit_refs = false` é o modo estrito: em posições de referência só
    /// são aceitos ponteiros válidos, então um id inventado fica irrepresentável.
    /// Exige que o banco contenha todas as entidades legítimas (prompt + TOOL).
    pub fn permits(self, sym: &Sym, bank_len: usize, allow_lit_refs: bool) -> bool {
        match (self, sym) {
            (Expect::Op, Sym::Op(_)) => true,
            (Expect::Lit, Sym::Lit(_)) => true,
            (Expect::Num, Sym::Num(_)) => true,
            (Expect::Ref | Expect::RefOrEnd, Sym::Ptr(k)) => (*k as usize) < bank_len,
            (Expect::Ref | Expect::RefOrEnd, Sym::Lit(_)) => allow_lit_refs,
            (Expect::RefOrEnd, Sym::End) => true,
            _ => false,
        }
    }
}

/// Dado um prefixo válido, diz o que pode vir a seguir.
pub fn expect(prefix: &[Sym]) -> Result<Expect, IsaError> {
    let mut pos = 0usize;
    while pos < prefix.len() {
        let op = match &prefix[pos] {
            Sym::Op(op) => *op,
            _ => return Err(IsaError::Unexpected { pos, want: "op" }),
        };
        pos += 1;
        for kind in op.shape() {
            match kind {
                Kind::Ref => match prefix.get(pos) {
                    None => return Ok(Expect::Ref),
                    Some(Sym::Ptr(_) | Sym::Lit(_)) => pos += 1,
                    Some(_) => {
                        return Err(IsaError::Unexpected {
                            pos,
                            want: "ptr ou literal",
                        });
                    }
                },
                Kind::Lit => match prefix.get(pos) {
                    None => return Ok(Expect::Lit),
                    Some(Sym::Lit(_)) => pos += 1,
                    Some(_) => {
                        return Err(IsaError::Unexpected {
                            pos,
                            want: "literal",
                        });
                    }
                },
                Kind::Num => match prefix.get(pos) {
                    None => return Ok(Expect::Num),
                    Some(Sym::Num(_)) => pos += 1,
                    Some(_) => {
                        return Err(IsaError::Unexpected {
                            pos,
                            want: "número",
                        });
                    }
                },
                Kind::RefList => loop {
                    match prefix.get(pos) {
                        None => return Ok(Expect::RefOrEnd),
                        Some(Sym::End) => {
                            pos += 1;
                            break;
                        }
                        Some(Sym::Ptr(_) | Sym::Lit(_)) => pos += 1,
                        Some(_) => {
                            return Err(IsaError::Unexpected {
                                pos,
                                want: "ptr, literal ou end",
                            });
                        }
                    }
                },
            }
        }
    }
    Ok(Expect::Op)
}

// Formato texto

pub fn to_text(syms: &[Sym]) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(syms.len());
    for s in syms {
        parts.push(match s {
            Sym::Op(op) => format!("<{}>", op.name()),
            Sym::Ptr(k) => format!("#{k}"),
            Sym::Num(n) => format!("%{n}"),
            Sym::End => "<END>".to_owned(),
            Sym::Lit(t) => {
                let mut q = String::with_capacity(t.len() + 2);
                q.push('"');
                for ch in t.chars() {
                    match ch {
                        '\\' => q.push_str("\\\\"),
                        '"' => q.push_str("\\\""),
                        '\n' => q.push_str("\\n"),
                        c => q.push(c),
                    }
                }
                q.push('"');
                q
            }
        });
    }
    parts.join(" ")
}

pub fn from_text(s: &str) -> Result<Vec<Sym>, IsaError> {
    let mut out = Vec::new();
    let mut it = s.chars().peekable();
    loop {
        while it.peek().is_some_and(|c| c.is_whitespace()) {
            it.next();
        }
        let Some(&c) = it.peek() else { break };
        if c == '"' {
            it.next();
            let mut lit = String::new();
            loop {
                match it.next() {
                    None => return Err(IsaError::Text("literal não terminado")),
                    Some('"') => break,
                    Some('\\') => match it.next() {
                        Some('\\') => lit.push('\\'),
                        Some('"') => lit.push('"'),
                        Some('n') => lit.push('\n'),
                        _ => return Err(IsaError::Text("escape inválido")),
                    },
                    Some(ch) => lit.push(ch),
                }
            }
            out.push(Sym::Lit(lit));
        } else {
            let mut w = String::new();
            while let Some(&ch) = it.peek() {
                if ch.is_whitespace() || ch == '"' {
                    break;
                }
                w.push(ch);
                it.next();
            }
            out.push(parse_word(&w)?);
        }
    }
    Ok(out)
}

fn parse_word(w: &str) -> Result<Sym, IsaError> {
    if w == "<END>" {
        return Ok(Sym::End);
    }
    if let Some(inner) = w.strip_prefix('<').and_then(|r| r.strip_suffix('>')) {
        return Op::from_name(inner)
            .map(Sym::Op)
            .ok_or(IsaError::Text("op desconhecida"));
    }
    if let Some(n) = w.strip_prefix('#') {
        return n
            .parse::<Slot>()
            .map(Sym::Ptr)
            .map_err(|_| IsaError::Text("slot inválido"));
    }
    if let Some(n) = w.strip_prefix('%') {
        return n
            .parse::<u8>()
            .map(Sym::Num)
            .map_err(|_| IsaError::Text("número inválido"));
    }
    Err(IsaError::Text("símbolo desconhecido"))
}

// JSON canônico e custo

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for ch in s.chars() {
        match ch {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            '\u{08}' => o.push_str("\\b"),
            '\u{0c}' => o.push_str("\\f"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn json_list(v: &[String]) -> String {
    let items: Vec<String> = v.iter().map(|s| json_str(s)).collect();
    format!("[{}]", items.join(","))
}

/// JSON no formato do `LlmAction` (verificado contra `serde_json` em `isa_adapter.rs`).
pub fn canonical_json(a: &Action) -> String {
    match a {
        Action::QueryFact { entity, relation } => format!(
            "{{\"action\":\"query_fact\",\"entity\":{},\"relation\":{}}}",
            json_str(entity),
            json_str(relation)
        ),
        Action::SearchFacts { query, limit } => format!(
            "{{\"action\":\"search_facts\",\"query\":{},\"limit\":{limit}}}",
            json_str(query)
        ),
        Action::TraverseGraph {
            start_entity,
            max_hops,
        } => format!(
            "{{\"action\":\"traverse_graph\",\"start_entity\":{},\"max_hops\":{max_hops}}}",
            json_str(start_entity)
        ),
        Action::EvalMath { expression } => {
            format!(
                "{{\"action\":\"eval_math\",\"expression\":{}}}",
                json_str(expression)
            )
        }
        Action::ProposeAnswer { text, cited_facts } => format!(
            "{{\"action\":\"propose_answer\",\"text\":{},\"cited_facts\":{}}}",
            json_str(text),
            json_list(cited_facts)
        ),
        Action::SearchContext { query, limit } => format!(
            "{{\"action\":\"search_context\",\"query\":{},\"limit\":{limit}}}",
            json_str(query)
        ),
        Action::ReadSpans { span_ids } => {
            format!(
                "{{\"action\":\"read_spans\",\"span_ids\":{}}}",
                json_list(span_ids)
            )
        }
        Action::CheckLogic { smtlib2_script } => format!(
            "{{\"action\":\"check_logic\",\"smtlib2_script\":{}}}",
            json_str(smtlib2_script)
        ),
        Action::RunWasmCheck {
            wasm_wat_source,
            expected_stdout,
        } => format!(
            "{{\"action\":\"run_wasm_check\",\"wasm_wat_source\":{},\"expected_stdout\":{}}}",
            json_str(wasm_wat_source),
            json_str(expected_stdout)
        ),
    }
}

/// Contador de tokens GROSSEIRO (sem tokenizer): palavras em pedaços de 6 chars,
/// pontuação em pares, `_` conta como pontuação. Troque pelo tokenizer World real.
pub fn approx_tokens(s: &str) -> usize {
    let (mut n, mut word, mut punct) = (0usize, 0usize, 0usize);
    for ch in s.chars() {
        if ch.is_alphanumeric() {
            n += punct.div_ceil(2);
            punct = 0;
            word += 1;
        } else if ch.is_whitespace() {
            n += word.div_ceil(6) + punct.div_ceil(2);
            word = 0;
            punct = 0;
        } else {
            n += word.div_ceil(6);
            word = 0;
            punct += 1;
        }
    }
    n + word.div_ceil(6) + punct.div_ceil(2)
}

/// Custo em passos de decode: Op/Ptr/Num/End = 1; Lit = tokens do texto + 1 (delimitador).
pub fn isa_cost(syms: &[Sym], count: &dyn Fn(&str) -> usize) -> usize {
    syms.iter()
        .map(|s| match s {
            Sym::Lit(t) => count(t) + 1,
            _ => 1,
        })
        .sum()
}

pub fn json_cost(a: &Action, count: &dyn Fn(&str) -> usize) -> usize {
    count(&canonical_json(a))
}

// Testes

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    const POOL: [&str; 14] = [
        "",
        "plano_001",
        "fact-preco-001",
        "preco_mensal",
        "49.15",
        "a \"quoted\" b",
        "back\\slash",
        "line\nbreak",
        "ação — ünïcode 日本",
        "(2 * 3.5) / (1 - 1)",
        "  spaces  ",
        "#12",
        "<END>",
        "%5",
    ];

    fn rand_string(r: &mut Rng, bank: &[String]) -> String {
        match r.below(10) {
            0..=3 if !bank.is_empty() => bank[r.below(bank.len())].clone(),
            0..=7 => POOL[r.below(POOL.len())].to_owned(),
            _ => {
                let cs: Vec<char> = "ab \"\\\n#%<>é_9".chars().collect();
                (0..r.below(8)).map(|_| cs[r.below(cs.len())]).collect()
            }
        }
    }

    fn rand_bank(r: &mut Rng) -> (Bank, Vec<String>) {
        let mut bank = Bank::new();
        let mut items = Vec::new();
        for _ in 0..r.below(8) {
            let s = POOL[r.below(POOL.len())];
            bank.intern(s);
            items.push(s.to_owned());
        }
        (bank, items)
    }

    fn rand_action(r: &mut Rng, items: &[String]) -> Action {
        let list = |r: &mut Rng| {
            (0..r.below(4))
                .map(|_| rand_string(r, items))
                .collect::<Vec<_>>()
        };
        match r.below(9) {
            0 => Action::QueryFact {
                entity: rand_string(r, items),
                relation: rand_string(r, items),
            },
            1 => Action::SearchFacts {
                query: rand_string(r, items),
                limit: r.below(256) as u8,
            },
            2 => Action::TraverseGraph {
                start_entity: rand_string(r, items),
                max_hops: r.below(256) as u8,
            },
            3 => Action::EvalMath {
                expression: rand_string(r, items),
            },
            4 => Action::ProposeAnswer {
                text: rand_string(r, items),
                cited_facts: list(r),
            },
            5 => Action::SearchContext {
                query: rand_string(r, items),
                limit: r.below(256) as u8,
            },
            6 => Action::ReadSpans { span_ids: list(r) },
            7 => Action::CheckLogic {
                smtlib2_script: rand_string(r, items),
            },
            _ => Action::RunWasmCheck {
                wasm_wat_source: rand_string(r, items),
                expected_stdout: rand_string(r, items),
            },
        }
    }

    #[test]
    fn roundtrip_structural_and_text() {
        let mut r = Rng(0xDEADBEEF);
        for _ in 0..5000 {
            let (bank, items) = rand_bank(&mut r);
            let a = rand_action(&mut r, &items);
            let syms = encode(&a, &bank);
            let (back, used) = decode_one(&syms, &bank).expect("decode");
            assert_eq!(back, a);
            assert_eq!(used, syms.len());
            let txt = to_text(&syms);
            assert_eq!(from_text(&txt).expect("from_text"), syms, "texto: {txt}");
        }
    }

    #[test]
    fn roundtrip_stream() {
        let mut r = Rng(42);
        for _ in 0..1000 {
            let (bank, items) = rand_bank(&mut r);
            let actions: Vec<Action> = (0..1 + r.below(6))
                .map(|_| rand_action(&mut r, &items))
                .collect();
            let syms: Vec<Sym> = actions.iter().flat_map(|a| encode(a, &bank)).collect();
            assert_eq!(decode_stream(&syms, &bank).expect("stream"), actions);
        }
    }

    #[test]
    fn bank_never_increases_cost() {
        let mut r = Rng(7);
        for _ in 0..3000 {
            let (bank, items) = rand_bank(&mut r);
            let a = rand_action(&mut r, &items);
            let with = isa_cost(&encode(&a, &bank), &approx_tokens);
            let without = isa_cost(&encode(&a, &Bank::new()), &approx_tokens);
            assert!(with <= without, "{a:?}: {with} > {without}");
        }
    }

    #[test]
    fn expect_never_forbids_the_true_continuation() {
        let mut r = Rng(99);
        for _ in 0..2000 {
            let (bank, items) = rand_bank(&mut r);
            let actions: Vec<Action> = (0..1 + r.below(3))
                .map(|_| rand_action(&mut r, &items))
                .collect();
            let syms: Vec<Sym> = actions.iter().flat_map(|a| encode(a, &bank)).collect();
            for i in 0..syms.len() {
                let e = expect(&syms[..i]).expect("prefixo válido");
                assert!(
                    e.permits(&syms[i], bank.len(), true),
                    "i={i} e={e:?} sym={:?}",
                    syms[i]
                );
            }
            assert_eq!(expect(&syms).expect("completo"), Expect::Op);
        }
    }

    #[test]
    fn strict_mode_makes_unknown_ids_unrepresentable() {
        assert!(!Expect::RefOrEnd.permits(&Sym::Lit("fact-alucinado-000".into()), 5, false));
        assert!(!Expect::RefOrEnd.permits(&Sym::Ptr(5), 5, false));
        assert!(Expect::RefOrEnd.permits(&Sym::Ptr(4), 5, false));
        assert!(Expect::RefOrEnd.permits(&Sym::End, 5, false));
    }

    #[test]
    fn bad_slot_and_garbage_never_panic() {
        let bank = Bank::new();
        assert_eq!(
            decode_one(&[Sym::Op(Op::ReadSpans), Sym::Ptr(3), Sym::End], &bank),
            Err(IsaError::BadSlot(3))
        );
        let mut r = Rng(1234);
        for _ in 0..20000 {
            let n = r.below(8);
            let syms: Vec<Sym> = (0..n)
                .map(|_| match r.below(5) {
                    0 => Sym::Op(Op::ALL[r.below(9)]),
                    1 => Sym::Ptr(r.below(4) as Slot),
                    2 => Sym::Lit(POOL[r.below(POOL.len())].to_owned()),
                    3 => Sym::Num(r.below(256) as u8),
                    _ => Sym::End,
                })
                .collect();
            let (b, _) = rand_bank(&mut r);
            let _ = decode_one(&syms, &b);
            let _ = decode_stream(&syms, &b);
            let _ = expect(&syms);
        }
        for _ in 0..20000 {
            let cs: Vec<char> = "<>#%\"\\ nENDREAQ0129".chars().collect();
            let s: String = (0..r.below(12)).map(|_| cs[r.below(cs.len())]).collect();
            let _ = from_text(&s);
        }
    }

    #[test]
    fn golden_read_spans() {
        let mut bank = Bank::new();
        bank.intern("cl_02");
        bank.intern("cl_47");
        let a = Action::ReadSpans {
            span_ids: vec!["cl_02".into(), "cl_47".into()],
        };
        let syms = encode(&a, &bank);
        assert_eq!(to_text(&syms), "<READ> #0 #1 <END>");
        assert_eq!(isa_cost(&syms, &approx_tokens), 4);
        assert!(json_cost(&a, &approx_tokens) >= 15);
    }
}
