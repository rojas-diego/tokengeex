use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use tokengeex::{
    lattice::{Lattice, VecPool},
    Model, ModelWrapper, Processor, TokenID, CAPCODE_RE,
};

fn load_samples() -> (Vec<String>, usize) {
    let data = std::fs::read("./data/train.bin").unwrap();
    let samples: Vec<String> = data
        .split(|&b| b == b'\0')
        .map(|s| String::from_utf8_lossy(s).to_string())
        .collect();
    let bytes = samples.iter().map(|s| s.len()).sum::<usize>();

    (samples, bytes)
}

fn load_processed_samples(processors: &[Box<dyn Processor>]) -> (Vec<String>, usize) {
    let (samples, _) = load_samples();
    let processed_samples = processors.iter().fold(samples, |samples, processor| {
        samples
            .iter()
            .map(|s| processor.preprocess(s))
            .collect::<Vec<_>>()
    });
    let bytes = processed_samples.iter().map(|s| s.len()).sum::<usize>();

    (processed_samples, bytes)
}

fn load_many_processed_samples(
    processors: &[Box<dyn Processor>],
    size: usize,
) -> (Vec<String>, usize) {
    let (samples, _) = load_processed_samples(processors);

    let samples = samples
        .iter()
        .cycle()
        .take(size)
        .cloned()
        .collect::<Vec<String>>();

    let bytes = samples.iter().map(|s| s.len()).sum::<usize>();

    (samples, bytes)
}

fn lattice(c: &mut Criterion) {
    let (mut samples, bytes) = load_many_processed_samples(
        &[
            Box::new(tokengeex::CapcodeProcessor),
            Box::new(tokengeex::CrlfProcessor),
            Box::new(tokengeex::UnicodeProcessor::Nfc),
        ],
        10000,
    );

    let mut group = c.benchmark_group("lattice");
    group.throughput(Throughput::Bytes(bytes as u64));
    group.sample_size(10);

    group.bench_function("from", |b| {
        b.iter(|| {
            let mut lattice = Lattice::default();
            let mut pool = VecPool::with_capacity(1024 * 128, 16);

            for sample in &samples {
                lattice.from(sample.as_bytes(), 0, 1, 0, &mut pool);
            }
        });
    });

    let tokenizer = tokengeex::load("./data/unigram-65k.json").unwrap();
    let ModelWrapper::Unigram(model) = tokenizer.model();

    println!(
        "Lattice tests are slow. Currently using {} threads.",
        rayon::current_num_threads()
    );

    group.bench_function("from_populate_nodes_multithreaded", |b| {
        b.iter(|| {
            let chunk_size = std::cmp::max(1, samples.len() / rayon::current_num_threads());
            let delete_token_id = model
                .token_to_id("D")
                .unwrap_or(model.vocab_size() as TokenID);

            samples.par_chunks(chunk_size).for_each(|chunk| {
                let mut lattice = Lattice::default();
                let mut pool = VecPool::with_capacity(1024 * 128, 16);

                for sample in chunk {
                    lattice.from(
                        sample.as_bytes(),
                        (model.vocab_size()) as TokenID,
                        (model.vocab_size() + 1) as TokenID,
                        delete_token_id,
                        &mut pool,
                    );
                    model.populate_nodes(&mut lattice);
                }
            });
        });
    });

    samples.truncate(1000);
    let bytes = samples.iter().map(|s| s.len()).sum::<usize>();

    group.throughput(Throughput::Bytes(bytes as u64));
    group.bench_function("from_populate_nodes_marginal_multithreaded", |b| {
        b.iter(|| {
            let chunk_size = std::cmp::max(1, samples.len() / rayon::current_num_threads());

            samples.par_chunks(chunk_size).for_each(|chunk| {
                let mut lattice = Lattice::default();
                let mut pool = VecPool::with_capacity(1024 * 128, 16);
                let mut frequencies = vec![0.0; model.vocab_size()];
                let delete_token_id = model
                    .token_to_id("D")
                    .unwrap_or(model.vocab_size() as TokenID);

                for sample in chunk {
                    lattice.from(
                        sample.as_bytes(),
                        (model.vocab_size()) as TokenID,
                        (model.vocab_size() + 1) as TokenID,
                        delete_token_id,
                        &mut pool,
                    );

                    model.populate_nodes(&mut lattice);

                    lattice.populate_marginal(&mut frequencies);
                }
            });
        });
    });

    group.finish();
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
    let (samples, bytes) = load_many_processed_samples(
        &[
            Box::new(tokengeex::CapcodeProcessor),
            Box::new(tokengeex::CrlfProcessor),
            Box::new(tokengeex::UnicodeProcessor::Nfc),
        ],
        1000,
    );

    let mut group = c.benchmark_group("vocabulary_generator");
    group.throughput(Throughput::Bytes(bytes as u64));
    group.sampling_mode(criterion::SamplingMode::Flat);
    group.sample_size(10);

    let mut generator = tokengeex::VocabularyGenerator::new(24, 0.01, CAPCODE_RE);
    group.bench_function("feed_capcode_24_1pct", |b| {
        b.iter(|| {
            generator.feed(samples.as_slice());
        });
    });

    group.finish();
}

fn tokenizer_unigram(c: &mut Criterion) {
    let (samples, bytes) = load_samples();
    let tokenizer = tokengeex::load("./data/unigram-65k.json").unwrap();

    let mut group = c.benchmark_group("tokenizer_unigram");
    group.confidence_level(0.95).sample_size(25);
    group.throughput(Throughput::Bytes(bytes as u64));

    group.bench_function("encode", |b| {
        b.iter(|| {
            for s in &samples {
                tokenizer.encode(s).unwrap();
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
    lattice,
    tokenizer_unigram
);
criterion_main!(bench);
