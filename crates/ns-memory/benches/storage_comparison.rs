use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ns_core::fact::{Fact, FactId};
use ns_core::ports::DeterministicMemory;
use ns_memory::RedbMemory;

const TOTAL_ENTITIES: usize = 20_000;
const RELS_PER_ENTITY: usize = 5;
const TOTAL_FACTS: usize = TOTAL_ENTITIES * RELS_PER_ENTITY; // 100_000 fatos

const COLD_LOOKUP_SAMPLES: usize = 10_000;
const BFS_SAMPLES: usize = 1_000;
const BFS_MAX_HOPS: u8 = 3;

const RELATIONS: [&str; RELS_PER_ENTITY] = [
    "depends_on",
    "relates_to",
    "category",
    "unit_price",
    "status",
];

struct TempBenchDir(PathBuf);

impl TempBenchDir {
    fn new() -> Result<Self, std::io::Error> {
        let dir = std::env::temp_dir().join(format!("neurosym-bench-{}", std::process::id()));
        if dir.exists() {
            fs::remove_dir_all(&dir)?;
        }
        fs::create_dir_all(&dir)?;
        Ok(Self(dir))
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempBenchDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Debug, Clone, Copy)]
struct LatencyStats {
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    ops_per_sec: f64,
}

impl LatencyStats {
    fn from_nanos(mut samples_ns: Vec<u128>, total_elapsed: Duration) -> Self {
        samples_ns.sort_unstable();
        let count = samples_ns.len().max(1);
        let sum_ns: u128 = samples_ns.iter().copied().sum();

        let p50_ns = percentile(&samples_ns, 50);
        let p95_ns = percentile(&samples_ns, 95);
        let p99_ns = percentile(&samples_ns, 99);

        let mean_us = (sum_ns as f64) / (count as f64) / 1_000.0;
        let ops_per_sec = (count as f64) / total_elapsed.as_secs_f64().max(1e-9);

        Self {
            mean_us,
            p50_us: (p50_ns as f64) / 1_000.0,
            p95_us: (p95_ns as f64) / 1_000.0,
            p99_us: (p99_ns as f64) / 1_000.0,
            ops_per_sec,
        }
    }
}

#[derive(Debug, Clone)]
struct EngineReport {
    name: &'static str,
    insert_elapsed: Duration,
    disk_bytes_raw: u64,
    disk_bytes_settled: u64,
    cold_lookup: LatencyStats,
    hot_ram_lookup: LatencyStats,
    bfs_traversal: LatencyStats,
    avg_bfs_facts: f64,
}

fn percentile(sorted: &[u128], pct: usize) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = sorted.len().saturating_sub(1).saturating_mul(pct) / 100;
    sorted[idx]
}

fn generate_dataset() -> Vec<Fact> {
    let mut facts = Vec::with_capacity(TOTAL_FACTS);

    for entity_idx in 0..TOTAL_ENTITIES {
        let subject = format!("entity_{entity_idx:05}");
        let next_entity = format!("entity_{:05}", (entity_idx + 1) % TOTAL_ENTITIES);
        let jump_entity = format!("entity_{:05}", (entity_idx + 7) % TOTAL_ENTITIES);

        let objects = [
            next_entity,
            jump_entity,
            format!("grupo_{:03}", entity_idx % 250),
            format!("{}.{:02}", 10 + (entity_idx % 990), entity_idx % 100),
            if entity_idx % 2 == 0 {
                "ativo".to_owned()
            } else {
                "revisao".to_owned()
            },
        ];

        for (rel_idx, relation) in RELATIONS.iter().enumerate() {
            let fact_seq = entity_idx
                .saturating_mul(RELS_PER_ENTITY)
                .saturating_add(rel_idx);
            facts.push(Fact {
                id: FactId::new(format!("f-{fact_seq:06}")),
                subject: subject.clone(),
                relation: (*relation).to_owned(),
                object: objects[rel_idx].clone(),
                source_ref: format!("dataset#shard-{}", entity_idx % 64),
            });
        }
    }

    facts
}

/// Gera `COLD_LOOKUP_SAMPLES` pares `(entity, relation)` garantidamente distintos
/// espalhados por toda a B-Tree via passo coprimo com `100_000`.
fn generate_cold_lookup_keys() -> Vec<(String, &'static str)> {
    let mut keys = Vec::with_capacity(COLD_LOOKUP_SAMPLES);
    for i in 0..COLD_LOOKUP_SAMPLES {
        let seq = (i.saturating_mul(79_193).saturating_add(17)) % TOTAL_FACTS;
        let entity_idx = seq / RELS_PER_ENTITY;
        let rel_idx = seq % RELS_PER_ENTITY;
        keys.push((format!("entity_{entity_idx:05}"), RELATIONS[rel_idx]));
    }
    keys
}

/// Gera `BFS_SAMPLES` entidades iniciais distintas espalhadas por `0..20_000`.
fn generate_bfs_starts() -> Vec<String> {
    let mut starts = Vec::with_capacity(BFS_SAMPLES);
    for i in 0..BFS_SAMPLES {
        let entity_idx = (i.saturating_mul(7_919).saturating_add(13)) % TOTAL_ENTITIES;
        starts.push(format!("entity_{entity_idx:05}"));
    }
    starts
}

fn file_len_or_zero(path: &Path) -> u64 {
    fs::metadata(path).map_or(0, |meta| meta.len())
}

async fn bench_lookups<M: DeterministicMemory>(
    memory: &M,
    keys: &[(String, &'static str)],
) -> Result<LatencyStats, Box<dyn std::error::Error>> {
    let mut samples_ns = Vec::with_capacity(keys.len());
    let total_start = Instant::now();

    for (entity, relation) in keys {
        let t0 = Instant::now();
        let found = memory.lookup_fact(entity, relation).await?;
        let elapsed_ns = t0.elapsed().as_nanos();

        if found.is_none() {
            return Err(format!("missing fact ({entity}, {relation}) during benchmark").into());
        }
        black_box(found);
        samples_ns.push(elapsed_ns);
    }

    Ok(LatencyStats::from_nanos(samples_ns, total_start.elapsed()))
}

async fn bench_bfs<M: DeterministicMemory>(
    memory: &M,
    starts: &[String],
) -> Result<(LatencyStats, f64), Box<dyn std::error::Error>> {
    let mut samples_ns = Vec::with_capacity(starts.len());
    let mut total_facts_visited: usize = 0;
    let total_start = Instant::now();

    for start in starts {
        let t0 = Instant::now();
        let facts = memory.traverse(start, BFS_MAX_HOPS).await?;
        let elapsed_ns = t0.elapsed().as_nanos();

        total_facts_visited = total_facts_visited.saturating_add(facts.len());
        black_box(facts);
        samples_ns.push(elapsed_ns);
    }

    let avg_facts = (total_facts_visited as f64) / (starts.len().max(1) as f64);
    Ok((
        LatencyStats::from_nanos(samples_ns, total_start.elapsed()),
        avg_facts,
    ))
}

async fn run_redb_benchmark(
    db_path: &Path,
    dataset: &[Fact],
    lookup_keys: &[(String, &'static str)],
    bfs_starts: &[String],
) -> Result<EngineReport, Box<dyn std::error::Error>> {
    // 1. Escrita em lote
    let (insert_elapsed, disk_bytes_raw) = {
        let mut writer = RedbMemory::open_file(db_path)?;
        let t0 = Instant::now();
        let inserted = writer.upsert_batch(dataset)?;
        let elapsed = t0.elapsed();
        assert_eq!(inserted, TOTAL_FACTS);
        let raw_size = file_len_or_zero(db_path);
        let _ = writer.compact()?;
        (elapsed, raw_size)
    };
    let disk_bytes_settled = file_len_or_zero(db_path);

    // 2. Leitura fria (reabre do zero: FactCache vazio + 10k chaves distintas)
    let (cold_lookup, hot_ram_lookup) = {
        let reader = RedbMemory::open_file(db_path)?;
        let cold = bench_lookups(&reader, lookup_keys).await?;
        let hot = bench_lookups(&reader, lookup_keys).await?;
        (cold, hot)
    };

    // 3. Travessia BFS (reabre instância limpa)
    let (bfs_traversal, avg_bfs_facts) = {
        let reader = RedbMemory::open_file(db_path)?;
        bench_bfs(&reader, bfs_starts).await?
    };

    Ok(EngineReport {
        name: "redb (B+Tree Rust)",
        insert_elapsed,
        disk_bytes_raw,
        disk_bytes_settled,
        cold_lookup,
        hot_ram_lookup,
        bfs_traversal,
        avg_bfs_facts,
    })
}

fn mib(bytes: u64) -> f64 {
    (bytes as f64) / (1024.0 * 1024.0)
}

fn print_report(redb: &EngineReport) {
    println!();
    println!("=========================================================================");
    println!(
        "  NeuroSym Storage Benchmark | {TOTAL_FACTS} fatos | {COLD_LOOKUP_SAMPLES} cold lookups | {BFS_SAMPLES} BFS ({BFS_MAX_HOPS} hops)"
    );
    println!("=========================================================================");
    println!("{:<34} | {:>24}", "Métrica", redb.name);
    println!("-----------------------------------+--------------------------");

    println!(
        "{:<34} | {:>21.2} ms",
        "Ingestão 100k fatos (batch)",
        redb.insert_elapsed.as_secs_f64() * 1_000.0,
    );
    println!(
        "{:<34} | {:>19.2} MiB",
        "Tamanho em disco (pós-carga)",
        mib(redb.disk_bytes_raw),
    );
    println!(
        "{:<34} | {:>19.2} MiB",
        "Tamanho estabilizado (compact)",
        mib(redb.disk_bytes_settled),
    );
    println!(
        "{:<34} | {:>18.1} B/fato",
        "Peso médio por fato (com índices)",
        (redb.disk_bytes_settled as f64) / (TOTAL_FACTS as f64),
    );
    println!("-----------------------------------+--------------------------");
    println!(
        "{:<34} | {:>21.2} us",
        "Cold lookup (média)", redb.cold_lookup.mean_us
    );
    println!(
        "{:<34} | {:>21.2} us",
        "Cold lookup (p50 / mediana)", redb.cold_lookup.p50_us
    );
    println!(
        "{:<34} | {:>21.2} us",
        "Cold lookup (p95)", redb.cold_lookup.p95_us
    );
    println!(
        "{:<34} | {:>21.2} us",
        "Cold lookup (p99)", redb.cold_lookup.p99_us
    );
    println!(
        "{:<34} | {:>18.0} ops/s",
        "Cold lookup (throughput)", redb.cold_lookup.ops_per_sec,
    );
    println!("-----------------------------------+--------------------------");
    println!(
        "{:<34} | {:>21.2} us",
        "Hot RAM lookup (DashMap p50)", redb.hot_ram_lookup.p50_us,
    );
    println!("-----------------------------------+--------------------------");
    println!(
        "{:<34} | {:>21.2} us",
        "Travessia BFS 3 hops (média)", redb.bfs_traversal.mean_us,
    );
    println!(
        "{:<34} | {:>21.2} us",
        "Travessia BFS 3 hops (p50)", redb.bfs_traversal.p50_us,
    );
    println!(
        "{:<34} | {:>21.2} us",
        "Travessia BFS 3 hops (p95)", redb.bfs_traversal.p95_us,
    );
    println!(
        "{:<34} | {:>21.2} us",
        "Travessia BFS 3 hops (p99)", redb.bfs_traversal.p99_us,
    );
    println!(
        "{:<34} | {:>18.0} ops/s",
        "Travessia BFS (throughput)", redb.bfs_traversal.ops_per_sec,
    );
    println!(
        "{:<34} | {:>18.1} fatos",
        "Média de fatos coletados / BFS", redb.avg_bfs_facts,
    );
    println!("=========================================================================");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let temp_dir = TempBenchDir::new()?;
    let dataset = generate_dataset();
    let lookup_keys = generate_cold_lookup_keys();
    let bfs_starts = generate_bfs_starts();

    let redb_path = temp_dir.path("bench_100k.redb");

    let redb_report = runtime.block_on(run_redb_benchmark(
        &redb_path,
        &dataset,
        &lookup_keys,
        &bfs_starts,
    ))?;

    print_report(&redb_report);
    Ok(())
}
