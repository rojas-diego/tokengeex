use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use tokengeex::Processor;

fn load_samples() -> (Vec<String>, usize) {
    let data = std::fs::read("./data/train.bin").unwrap();
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
    group.confidence_level(0.95).sample_size(25);
    group.throughput(Throughput::Bytes(bytes as u64));

    group.bench_function("preprocess", |b| {
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
    group.throughput(Throughput::Bytes(bytes as u64));

    group.bench_function("postprocess", |b| {
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
    group.confidence_level(0.95).sample_size(25);
    group.throughput(Throughput::Bytes(bytes as u64));

    group.bench_function("preprocess", |b| {
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

    group.throughput(Throughput::Bytes(bytes as u64));
    group.bench_function("postprocess", |b| {
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
    group.confidence_level(0.95).sample_size(25);
    group.throughput(Throughput::Bytes(bytes as u64));

    group.bench_function("preprocess", |b| {
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

    group.throughput(Throughput::Bytes(bytes as u64));
    group.bench_function("postprocess", |b| {
        b.iter(|| {
            for s in &processed_samples {
                processor.postprocess(s);
            }
        });
    });
    group.finish();
}

fn vocabulary_generator(c: &mut Criterion) {
    let (samples, _) = load_samples();
    let samples = samples
        .iter()
        .map(|s| tokengeex::CapcodeProcessor.preprocess(s))
        .map(|s| tokengeex::CrlfProcessor.preprocess(&s))
        .map(|s| tokengeex::UnicodeProcessor::Nfc.preprocess(&s))
        .collect::<Vec<_>>();
    let bytes = samples.iter().map(|s| s.len()).sum::<usize>();

    let generator = tokengeex::VocabularyGenerator::new(
        24,
        0.001,
        &[r#"^(?:.|\s|[[:punct:][:\s:]]*[DUC]{0,2}[[:punct:][:\s:]]*| ?(?:[DUC]+) ?| ?[a-z]+(?: [a-z]+){0,2}| ?[0-9]{1,3})$"#.to_string()],
        &[],
    );

    let mut group = c.benchmark_group("vocabulary_generator");
    group.confidence_level(0.95);
    group.throughput(Throughput::Bytes(bytes as u64));

    group.bench_function("vocabulary_generator", |b| {
        b.iter(|| generator.collect_frequent_tokens(samples.iter().map(|s| s.as_str())));
    });

    group.finish();
}

fn tokenizer_unigram_encode(c: &mut Criterion) {
    let (samples, bytes) = load_samples();
    let tokenizer = tokengeex::load("./benches/unigram.json").unwrap();

    let mut group = c.benchmark_group("tokenizer_unigram_encode");
    group.confidence_level(0.95).sample_size(25);
    group.throughput(Throughput::Bytes(bytes as u64));

    group.bench_function("unigram_encode", |b| {
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
    vocabulary_generator,
    tokenizer_unigram_encode
);
criterion_main!(bench);
