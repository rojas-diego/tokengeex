use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn capcode_encode(c: &mut Criterion) {
    let data = std::fs::read("./benches/data.bin").unwrap();
    let samples: Vec<_> = data
        .split(|&b| b == b'\0')
        .map(|s| String::from_utf8_lossy(s))
        .collect();

    let total_bytes = samples.iter().map(|s| s.len()).sum::<usize>();

    let mut group = c.benchmark_group("capcode_encode");
    group.throughput(Throughput::Bytes(total_bytes as u64));
    group.bench_function(BenchmarkId::new("encode", total_bytes), |b| {
        b.iter(|| {
            for s in &samples {
                tokengeex::capcode::encode(s);
            }
        });
    });
    group.finish();
}

fn tokenizer_unigram_encode(c: &mut Criterion) {
    let data = std::fs::read("./benches/data.bin").unwrap();
    let samples: Vec<_> = data
        .split(|&b| b == b'\0')
        .map(|s| String::from_utf8_lossy(s))
        .collect();

    let total_bytes = samples.iter().map(|s| s.len()).sum::<usize>();

    let tokenizer = tokengeex::load("./benches/unigram.json").unwrap();

    let mut group = c.benchmark_group("tokenizer_unigram_encode");
    group.throughput(Throughput::Bytes(total_bytes as u64));
    group.bench_function(BenchmarkId::new("encode", total_bytes), |b| {
        b.iter(|| {
            for s in &samples {
                tokenizer.encode(s);
            }
        });
    });
    group.finish();
}

criterion_group!(bench, capcode_encode, tokenizer_unigram_encode);
criterion_main!(bench);
