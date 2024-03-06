use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tokengeex::Processor;

fn load_samples() -> (Vec<String>, usize) {
    let data = std::fs::read("./benches/1MB.bin").unwrap();
    let samples: Vec<String> = data
        .split(|&b| b == b'\0')
        .map(|s| String::from_utf8_lossy(s).to_string())
        .collect();
    let bytes = samples.iter().map(|s| s.len()).sum::<usize>();

    (samples, bytes)
}

fn processor_capcode(c: &mut Criterion) {
    let (samples, bytes) = load_samples();
    let processor = tokengeex::CapcodeProcessor;

    let mut group = c.benchmark_group("processor_capcode");
    group.throughput(Throughput::Bytes(bytes as u64));
    group.bench_function(BenchmarkId::new("preprocess", bytes), |b| {
        b.iter(|| {
            for s in &samples {
                processor.preprocess(s);
            }
        });
    });

    let processed_samples = samples
        .iter()
        .map(|s| processor.preprocess(s))
        .collect::<Vec<_>>();
    let bytes = processed_samples.iter().map(|s| s.len()).sum::<usize>();

    group.bench_function(BenchmarkId::new("postprocess", bytes), |b| {
        b.iter(|| {
            for s in &processed_samples {
                processor.postprocess(s);
            }
        });
    });
    group.finish();
}

fn processor_crlf(c: &mut Criterion) {
    let (samples, bytes) = load_samples();
    let processor = tokengeex::CrlfProcessor;

    let mut group = c.benchmark_group("processor_crlf");
    group.throughput(Throughput::Bytes(bytes as u64));
    group.bench_function(BenchmarkId::new("preprocess", bytes), |b| {
        b.iter(|| {
            for s in &samples {
                processor.preprocess(s);
            }
        });
    });

    let processed_samples = samples
        .iter()
        .map(|s| processor.preprocess(s))
        .collect::<Vec<_>>();
    let bytes = processed_samples.iter().map(|s| s.len()).sum::<usize>();

    group.bench_function(BenchmarkId::new("postprocess", bytes), |b| {
        b.iter(|| {
            for s in &processed_samples {
                processor.postprocess(s);
            }
        });
    });
    group.finish();
}

fn processor_unicode(c: &mut Criterion) {
    let (samples, bytes) = load_samples();
    let processor = tokengeex::UnicodeProcessor::Nfc;

    let mut group = c.benchmark_group("processor_unicode");
    group.throughput(Throughput::Bytes(bytes as u64));
    group.bench_function(BenchmarkId::new("preprocess", bytes), |b| {
        b.iter(|| {
            for s in &samples {
                processor.preprocess(s);
            }
        });
    });

    let processed_samples = samples
        .iter()
        .map(|s| processor.preprocess(s))
        .collect::<Vec<_>>();
    let bytes = processed_samples.iter().map(|s| s.len()).sum::<usize>();

    group.bench_function(BenchmarkId::new("postprocess", bytes), |b| {
        b.iter(|| {
            for s in &processed_samples {
                processor.postprocess(s);
            }
        });
    });
    group.finish();
}

fn tokenizer_unigram_encode(c: &mut Criterion) {
    let (samples, bytes) = load_samples();
    let tokenizer = tokengeex::load("./benches/unigram.json").unwrap();

    let mut group = c.benchmark_group("tokenizer_unigram_encode");
    group.throughput(Throughput::Bytes(bytes as u64));
    group.bench_function(BenchmarkId::new("unigram_encode", bytes), |b| {
        b.iter(|| {
            for s in &samples {
                tokenizer.encode(s);
            }
        });
    });
    group.finish();
}

criterion_group!(
    bench,
    processor_capcode,
    processor_crlf,
    processor_unicode,
    tokenizer_unigram_encode
);
criterion_main!(bench);
