use fnv::FnvHashMap;
use memmap2::Mmap;
use rayon::current_num_threads;
use serde::Serialize;
use std::{
    collections::{HashMap, HashSet},
    fs::{File, OpenOptions},
    io::Write,
    sync::RwLock,
};
use tokengeex::{
    parallelism::{MaybeParallelRefIterator, MaybeParallelSlice},
    task::{par_chunk_size, Task},
    unigram, CapcodeProcessor, CrlfProcessor, Model, Processor, ProcessorWrapper, TokenID,
    Tokenizer, UnicodeProcessor, VocabularyGenerator,
};

mod flags {
    xflags::xflags! {
        cmd tokengeex {
            /// Train a new tokeniser from data.
            cmd train {
                // --- General Purpose ---
                /// Kind of model to train.
                required -m, --model model: String
                /// Output tokeniser filepath.
                required -o, --output output: String
                /// Desired vocabulary size.
                required -v, --vocab-size vocab_size: usize
                /// Log filepath.
                required -l, --logfile logfile: String

                // --- Data ---
                /// List of source files to train the tokenizer on. Must be
                /// formatted according to {name}:{path}.
                repeated --train input: String
                /// List of source files to test the tokenizer on. Must be
                /// formatted according to {name}:{path}.
                repeated --test test: String

                // --- Processing ---
                /// Apply a processor to the input data.
                repeated --processor processor: String

                // --- Suggested, Added and Special Tokens ---
                /// Suggested token file.
                repeated --suggested-tokens-file suggested_tokens_file: String
                /// Added token file.
                repeated --added-tokens-file added_tokens_file: String
                /// Special token.
                repeated --special-token special_token: String

                // --- Initial Vocab ---
                /// The size of the initial vocabulary.
                optional --initial-vocab-size initial_vocab_size: usize
                /// Probability of inserting a new token to the vocabulary.
                optional --initial-vocab-insert-probability initial_vocab_insert_probability: f64
                /// Substrings that match this Regex will be considered for
                /// the vocabulary.
                optional --initial-vocab-allow initial_vocab_allow: String
                /// Maximum token length.
                optional --initial-vocab-max-token-length initial_vocab_max_token_length: usize

                // --- Unigram ---
                /// How much to shrink the vocabulary at each iteration.
                optional --unigram-shrinking-factor unigram_shrinking_factor: f64
                /// Number of sub-iterations for the EM algorithm.
                optional --unigram-num-sub-iterations unigram_num_sub_iterations: usize
                /// What kind of sample regularization to employ. Choices are
                /// "none", "log" or "constant". Defaults to "none".
                optional --unigram-sample-regularization unigram_sample_regularization: String
            }

            /// Improve a tokeniser using BPE.
            cmd bpe {
                /// Output tokeniser filepath.
                required -o, --output output: String
                /// Tokeniser vocabulary filepath.
                required -v, --vocab vocab: String
                /// The number of merge operations to perform.
                required -n, --num-merges num_merges: usize

                /// List of source files to train the tokenizer on. Must be
                /// formatted according to {name}:{path}.
                repeated --train train: String
                /// List of source files to test the tokenizer on. Must be
                /// formatted according to {name}:{path}.
                repeated --test test: String
                /// Step size for the BPE merge operations.
                optional --step step: usize
                /// Score scale factor.
                optional --score-scale-factor score_scale_factor: f64
                /// Max merge length
                optional --max-merge-length max_merge_length: usize
                /// Merges that match this Regex will be ignored.
                optional --ignore ignore: String
            }

            /// Evaluate the tokenizer on a test set.
            cmd evaluate {
                /// Tokeniser vocabulary filepath.
                required -v, --vocab vocab: String
                /// Log filepath.
                required -l, --logfile logfile: String
                /// List of source files to evaluate the tokenizer on. Must be
                /// formatted according to {name}:{path}.
                repeated --test test: String
            }

            /// Encode text using a tokeniser.
            cmd encode {
                /// Tokeniser vocabulary filepath.
                required -v, --vocab vocab: String
                /// Input text. Otherwise, stdin is used.
                optional -i, --input input: String
            }

            /// Decode tokenised IDs.
            cmd decode {
                /// Tokeniser vocabulary filepath.
                required -v, --vocab vocab: String
                /// Comma separated list of token IDs. Otherwise, stdin is used.
                optional -i, --input input: String
            }
        }
    }
}

/// Encode a sample using a tokeniser.
fn encode(input: Option<&str>, vocab: &str) {
    let input = input.map_or_else(
        || std::io::read_to_string(&mut std::io::stdin()).unwrap(),
        |s| s.to_string(),
    );

    let tokenizer = tokengeex::load(vocab).unwrap();

    let encoded = tokenizer.encode(&input).unwrap();

    let colors = vec![
        "\x1B[102m", // Bright Green background
        "\x1B[103m", // Bright Yellow background
        "\x1B[104m", // Bright Blue background
    ];

    let encoded = encoded
        .iter()
        .map(|id| {
            tokenizer
                .id_to_token(*id)
                .map(|(s, _)| String::from_utf8_lossy(&s).into())
                .unwrap()
        })
        .collect::<Vec<String>>();

    for (i, token) in encoded.iter().enumerate() {
        print!("{}{}", colors[i % colors.len()], token);
    }

    print!("\x1B[49m");
}

/// Decode a tokenised array of IDs.
fn decode(input: Option<&str>, vocab: &str) {
    let input = input.map_or_else(
        || std::io::read_to_string(&mut std::io::stdin()).unwrap(),
        |s| s.to_string(),
    );

    let tokenizer = tokengeex::load(vocab).unwrap();

    let decoded = tokenizer
        .decode(
            &input
                .split(',')
                .map(|s| s.parse().unwrap())
                .collect::<Vec<_>>(),
            true,
        )
        .unwrap();

    println!("{}", decoded);
}

fn load_processors(processors: &[String]) -> Vec<ProcessorWrapper> {
    processors
        .iter()
        .map(|name| {
            let processor = match name.as_str() {
                "capcode" => ProcessorWrapper::Capcode(CapcodeProcessor),
                "crlf" => ProcessorWrapper::Crlf(CrlfProcessor),
                "nfc" => ProcessorWrapper::Unicode(UnicodeProcessor::Nfc),
                "nfd" => ProcessorWrapper::Unicode(UnicodeProcessor::Nfd),
                "nfkc" => ProcessorWrapper::Unicode(UnicodeProcessor::Nfkc),
                "nfkd" => ProcessorWrapper::Unicode(UnicodeProcessor::Nfkd),
                _ => panic!("Processor {:?} is not supported.", name),
            };

            log::info!("Using processor {:?}", name);

            processor
        })
        .collect()
}

fn load_tokens_files(files: &[String], mode: &str) -> HashSet<String> {
    files
        .iter()
        .flat_map(|filename| {
            let file = std::fs::read_to_string(filename)
                .unwrap_or_else(|_| panic!("Could not read tokens file {:?}", filename));

            let tokens: Vec<String> = serde_json::from_str(&file)
                .unwrap_or_else(|_| panic!("Could not parse tokens file {:?}", filename));

            log::info!(
                "Loaded {} {} tokens from {:?}",
                tokens.len(),
                mode,
                filename
            );

            tokens
        })
        .collect()
}

struct Source {
    pub name: String,
    pub processed_samples: Vec<String>,
    #[allow(dead_code)]
    pub total_bytes: usize,
    pub total_chars: usize,
    #[allow(dead_code)]
    pub mmap: Mmap,
}

fn load_sources(sources: &[String], processors: &[ProcessorWrapper]) -> Vec<Source> {
    sources
        .maybe_par_iter()
        .map(|source| {
            let pieces = source.split(':').collect::<Vec<&str>>();
            if pieces.len() != 2 {
                panic!(
                    "Invalid source format: {:?}. Expected to be formatted as {{name}}:{{path}}",
                    source
                );
            }
            let name = pieces[0];
            let filepath = pieces[1];

            let file = std::fs::File::open(filepath).unwrap_or_else(|e| {
                panic!("Failed to open {:?}: {:?}", filepath, e);
            });

            let mmap = unsafe {
                Mmap::map(&file)
                    .unwrap_or_else(|e| panic!("Failed to mmap {:?}: {:?}", filepath, e))
            };

            let samples: Vec<&str> = mmap
                .split(|&b| b == 0x00)
                .map(|s| {
                    std::str::from_utf8(s).unwrap_or_else(|e| {
                        panic!("Sample in {:?} is not valid UTF-8: {:?}", filepath, e)
                    })
                })
                .filter(|s| !s.is_empty())
                .collect();

            let total_bytes = samples.iter().map(|s| s.len()).sum::<usize>();
            let total_chars = samples.iter().map(|s| s.chars().count()).sum::<usize>();

            let processed_samples: Vec<String> = samples
                .iter()
                .map(|&s| {
                    let mut sample = s.to_string();
                    for processor in processors {
                        sample = processor.preprocess(&sample);
                    }
                    sample
                })
                .filter(|s| !s.is_empty())
                .collect();

            let processed_total_bytes = processed_samples.iter().map(|s| s.len()).sum::<usize>();

            log::info!(
                "Loaded {:?} source from {:?} ({}). Samples: {} ({}). Processed Samples: {} ({}).",
                name,
                filepath,
                format_bytes_as_mb(mmap.len() as u64),
                samples.len(),
                format_bytes_as_mb(total_bytes as u64),
                processed_samples.len(),
                format_bytes_as_mb(processed_total_bytes as u64),
            );

            Source {
                name: name.to_string(),
                processed_samples,
                total_bytes,
                total_chars,
                mmap,
            }
        })
        .collect()
}

fn format_bytes_as_mb(bytes: u64) -> String {
    format!("{:.2}MB", bytes as f64 / 1_000_000.0)
}

struct FileLogger {
    file: File,
}

impl FileLogger {
    pub fn new(filepath: &str) -> Self {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(filepath)
            .unwrap_or_else(|e| panic!("Failed to create {:?}: {:?}", filepath, e));

        Self { file }
    }

    /// Write a serializable to the file as JSON.
    pub fn write<T: Serialize>(&mut self, data: &T) {
        let mut json =
            serde_json::to_string(data).unwrap_or_else(|e| panic!("Failed to serialize: {:?}", e));

        json.push('\n');

        self.file.write_all(json.as_bytes()).unwrap_or_else(|e| {
            panic!("Failed to write to {:?}: {:?}", self.file, e);
        });
    }
}

/// Train a new tokeniser from data.
#[allow(clippy::too_many_arguments)]
fn train(
    // --- General Purpose ---
    model: &str,
    output: &str,
    vocab_size: usize,
    logfile: &str,
    // --- Data ---
    train: &Vec<String>,
    test: &Vec<String>,
    // --- Processing ---
    processors: &Vec<String>,
    // --- Suggested, Added and Special Tokens ---
    suggested_tokens_files: &Vec<String>,
    added_tokens_files: &Vec<String>,
    special_tokens: &Vec<String>,
    // --- Initial Vocabulary ---
    initial_vocab_size: usize,
    initial_vocab_max_token_length: usize,
    initial_vocab_insert_probability: f64,
    initial_vocab_allow: &str,
    // --- Unigram ---
    unigram_shrinking_factor: f64,
    unigram_num_sub_iterations: usize,
    unigram_sample_regularization: &str,
) {
    assert!(
        train.len() > 0,
        "At least one training dataset must be provided"
    );

    log::info!("Writing logs to {:?}.", logfile);

    let mut logfile = FileLogger::new(logfile);

    let processors = load_processors(processors);
    let added_tokens = load_tokens_files(added_tokens_files, "added");
    let suggested_tokens = load_tokens_files(suggested_tokens_files, "suggested");

    log::info!(
        "Loaded {} added tokens and {} suggested tokens.",
        added_tokens.len(),
        suggested_tokens.len()
    );

    let train = load_sources(&train, &processors);
    let test = load_sources(&test, &processors);

    log::info!(
        "Training {:?} model with {} vocabulary entries. Writing to {:?}.",
        model,
        vocab_size,
        output,
    );

    match model {
        "unigram" => {
            log::info!(
                "Generating initial vocabulary of size {} (max_token_length: {}, insert_probability: {}).",
                initial_vocab_size, initial_vocab_max_token_length, initial_vocab_insert_probability
            );

            log::info!("Using allow rule: {:?}", initial_vocab_allow);

            let mut vocab_generator = VocabularyGenerator::new(
                initial_vocab_max_token_length,
                initial_vocab_insert_probability,
                initial_vocab_allow,
            );

            for source in &train {
                log::info!("Collecting frequent tokens from {:?}.", source.name);

                vocab_generator.feed(&source.processed_samples);

                log::info!(
                    "Collected frequent tokens from {:?}. Total: {}",
                    source.name,
                    vocab_generator.current_size()
                );
            }

            let (vocab, keep) =
                vocab_generator.generate(initial_vocab_size, &suggested_tokens, &added_tokens);
            let vocab_total_bytes = vocab.iter().map(|(s, _)| s.len()).sum::<usize>() as u64;

            log::info!(
                "Generated initial vocabulary of size {} ({}).",
                vocab.len(),
                format_bytes_as_mb(vocab_total_bytes as u64)
            );

            let mut model = unigram::Unigram::from(vocab, false);

            log::info!(
                "Training unigram model. vocab_size={} shrinking_factor={} num_sub_iterations={} sample_regularization={:?}",
                vocab_size,
                unigram_shrinking_factor,
                unigram_num_sub_iterations,
                unigram_sample_regularization,
            );
            let mut trainer = unigram::UnigramTrainer::new(
                vocab_size,
                unigram_num_sub_iterations,
                unigram_shrinking_factor,
                match unigram_sample_regularization {
                    "none" => unigram::SampleRegularization::None,
                    "log" => unigram::SampleRegularization::Log,
                    "consant" => unigram::SampleRegularization::Constant,
                    _ => panic!(
                        "Invalid sample regularization: {:?}",
                        unigram_sample_regularization
                    ),
                },
            );

            let all_train_samples = train
                .iter()
                .flat_map(|source| source.processed_samples.iter())
                .map(|s| s.as_str())
                .collect::<Vec<&str>>();

            let mut epoch = 0;
            let mut should_continue = true;
            while should_continue {
                log::info!("Epoch {} | Vocabulary size: {}", epoch, model.vocab_size());

                should_continue = trainer
                    .train(&mut model, &all_train_samples, &keep)
                    .unwrap();

                evaluate_impl(&mut logfile, "train", epoch, &train, &model);
                evaluate_impl(&mut logfile, "test", epoch, &test, &model);

                epoch += 1;
            }

            log::info!("Training finished. Writing to {:?}.", output);

            let mut tokenizer = Tokenizer::new(tokengeex::ModelWrapper::Unigram(model), processors);

            tokenizer.add_special_tokens(special_tokens);
            tokenizer.save(output).unwrap();
        }
        _ => {
            panic!("Model {:?} is not supported.", model);
        }
    }
}

#[derive(Serialize, Clone)]
struct Compression {
    pub num_tokens: usize,
    pub num_chars: usize,
    pub chars_per_token: f64,
}

#[derive(Serialize)]
struct Evaluation {
    pub epoch: usize,
    pub split: String,
    pub vocab_size: usize,
    pub compression: HashMap<String, Compression>,
    pub frequency_buckets: Vec<usize>,
}

fn evaluate_impl(
    logfile: &mut FileLogger,
    split: &str,
    epoch: usize,
    sources: &Vec<Source>,
    model: &unigram::Unigram,
) -> Evaluation {
    let token_frequencies = RwLock::new(vec![0; model.vocab_size()]);
    let mut compression = HashMap::new();

    for source in sources {
        let total_tokens = source
            .processed_samples
            .maybe_par_iter()
            .map(|s| {
                let ids = model.encode(s).unwrap();

                {
                    let mut token_frequencies = token_frequencies.write().unwrap();
                    ids.iter().for_each(|id| {
                        token_frequencies[*id as usize] += 1;
                    })
                }

                ids.len()
            })
            .sum::<usize>();

        let chars_per_token =
            ((source.total_chars as f64 / total_tokens as f64) * 100.0).round() / 100.0;

        let compression_for_source = Compression {
            num_chars: source.total_chars,
            num_tokens: total_tokens,
            chars_per_token,
        };

        compression.insert(source.name.clone(), compression_for_source.clone());

        log::info!(
            "{:>5} | {:>18} | {:>10} chars | {:>10} tokens | {:<4} chars per token",
            split.to_uppercase(),
            format!("{:?}", source.name),
            compression_for_source.num_chars,
            compression_for_source.num_tokens,
            compression_for_source.chars_per_token,
        );
    }

    let mut token_frequencies = token_frequencies
        .into_inner()
        .unwrap()
        .into_iter()
        .collect::<Vec<_>>();
    token_frequencies.sort_unstable_by(|a, b| b.cmp(a));

    let num_buckets = 50;
    let bucket_capacity = token_frequencies.len() / num_buckets;
    let mut frequency_buckets = vec![0usize; num_buckets];
    let mut current_bucket;

    for (i, &frequency) in token_frequencies.iter().enumerate() {
        current_bucket = i / bucket_capacity;
        current_bucket = current_bucket.min(num_buckets - 1);
        frequency_buckets[current_bucket] += frequency;
    }

    let evaluation = Evaluation {
        epoch,
        split: split.into(),
        vocab_size: model.vocab_size(),
        compression,
        frequency_buckets,
    };

    logfile.write(&evaluation);

    evaluation
}

fn evaluate(vocab: &str, logfile: &str, test: &Vec<String>) {
    let mut logfile = FileLogger::new(logfile);

    let tokenizer = tokengeex::load(vocab).unwrap();

    let test = load_sources(&test, &tokenizer.processors());

    let model = match tokenizer.model() {
        tokengeex::ModelWrapper::Unigram(unigram) => unigram,
    };

    evaluate_impl(&mut logfile, "test", 0, &test, &model);
}

fn bpe(
    output: &str,
    vocab: &str,
    num_merges: usize,
    train: &Vec<String>,
    step: usize,
    score_scale_factor: f64,
    max_merge_length: usize,
    ignore: &str,
) {
    log::info!(
        "BPE | merges={} step={} score_scale_factor={} max_merge_length={} ignore={:?}",
        num_merges,
        step,
        score_scale_factor,
        max_merge_length,
        ignore
    );

    let ignore = regex::Regex::new(ignore).unwrap();
    let mut tokenizer = tokengeex::load(vocab).unwrap();
    let mut logfile = FileLogger::new("/dev/null");

    let test = load_sources(&train, &tokenizer.processors());
    let train = load_sources(&train, &[]);

    let samples = train
        .iter()
        .flat_map(|source| source.processed_samples.iter())
        .map(|s| s.as_str())
        .collect::<Vec<&str>>();

    let baseline = {
        let model = match tokenizer.model() {
            tokengeex::ModelWrapper::Unigram(unigram) => unigram,
        };
        evaluate_impl(&mut logfile, "test", 0, &test, &model)
    };

    let mut ignored_pairs = FnvHashMap::<(TokenID, TokenID), usize>::default();

    for merges_completed in (0..num_merges).step_by(step) {
        let mut merges = std::cmp::min(step, num_merges - merges_completed);

        let chunk_size = par_chunk_size(samples.len(), current_num_threads() * 128, 10);
        let task = Task::new("BPE Merge", samples.len(), chunk_size);
        let pair_frequencies = RwLock::new(FnvHashMap::<(TokenID, TokenID), usize>::default());

        samples.maybe_par_chunks(chunk_size).for_each(|chunk| {
            let mut ltask = task.local(chunk.len());
            let mut local_pair_frequencies = FnvHashMap::<(TokenID, TokenID), usize>::default();

            for sample in chunk {
                let ids = tokenizer.encode(sample).unwrap();

                for i in 1..ids.len() {
                    let pair = (ids[i - 1], ids[i]);
                    *local_pair_frequencies.entry(pair).or_insert(0) += 1;
                }

                ltask.record(sample.len());
            }

            {
                let mut pair_frequencies = pair_frequencies.write().unwrap();
                for (pair, freq) in local_pair_frequencies {
                    *pair_frequencies.entry(pair).or_insert(0) += freq;
                }
            }

            ltask.finish();
        });

        let mut pairs = pair_frequencies
            .into_inner()
            .unwrap()
            .into_iter()
            .collect::<Vec<((TokenID, TokenID), usize)>>();

        pairs.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        for ((a, b), freq) in pairs.iter().copied() {
            if merges == 0 {
                break;
            }

            let pair = (a, b);

            let a = tokenizer.model_mut().vocab()[a as usize].clone();
            let b = tokenizer.model_mut().vocab()[b as usize].clone();

            let mut token = a.0.clone();
            token.extend_from_slice(&b.0);
            let score = (a.1 + b.1) * score_scale_factor;

            if token.len() > max_merge_length || ignore.is_match(&String::from_utf8_lossy(&token)) {
                if !ignored_pairs.contains_key(&pair) {
                    log::warn!(
                        "Skipped | {:<10} | {:<10} | {:<32} | Freq {:>8} | Score {:.2} | {}",
                        format!(
                            "{:?}",
                            String::from_utf8_lossy(&a.0[..std::cmp::min(8, a.0.len())])
                        ),
                        format!(
                            "{:?}",
                            String::from_utf8_lossy(&b.0[..std::cmp::min(8, b.0.len())])
                        ),
                        format!("{:?}", String::from_utf8_lossy(&token)),
                        freq,
                        score,
                        if token.len() > max_merge_length {
                            "TOO LONG"
                        } else {
                            "IGNORED"
                        }
                    );
                    ignored_pairs.insert(pair, freq);
                }
                continue;
            }

            tokenizer.add_tokens([(token.clone(), score)]);

            merges -= 1;

            log::info!(
                "Merged  | {:<10} | {:<10} | {:<32} | Freq {:>8} | Score {:.2}",
                format!(
                    "{:?}",
                    String::from_utf8_lossy(&a.0[..std::cmp::min(8, a.0.len())])
                ),
                format!(
                    "{:?}",
                    String::from_utf8_lossy(&b.0[..std::cmp::min(8, b.0.len())])
                ),
                format!("{:?}", String::from_utf8_lossy(&token)),
                freq,
                score
            );
        }

        let eval = {
            let model = match tokenizer.model() {
                tokengeex::ModelWrapper::Unigram(unigram) => unigram,
            };
            evaluate_impl(&mut logfile, "test", merges_completed, &test, &model)
        };

        // Compare the new evaluation with the previous one.
        let mut total_chars = 0;
        let mut baseline_total_tokens = 0;
        let mut new_total_tokens = 0;
        for source in eval.compression.keys() {
            let baseline_compression = baseline.compression.get(source).unwrap();
            let new_compression = eval.compression.get(source).unwrap();

            total_chars += new_compression.num_chars;
            baseline_total_tokens += baseline_compression.num_tokens;
            new_total_tokens += new_compression.num_tokens;

            log::info!(
                "DELTA | {:>18} | {:<4} -> {:<4} | +{:04.2} | +{:05.2}%",
                format!("{:?}", source),
                baseline_compression.chars_per_token,
                new_compression.chars_per_token,
                new_compression.chars_per_token - baseline_compression.chars_per_token,
                ((new_compression.chars_per_token - baseline_compression.chars_per_token)
                    / baseline_compression.chars_per_token
                    * 100.0)
            );
        }

        let total_chars_per_token =
            ((total_chars as f64 / new_total_tokens as f64) * 100.0).round() / 100.0;
        let baseline_chars_per_token =
            ((total_chars as f64 / baseline_total_tokens as f64) * 100.0).round() / 100.0;

        log::info!(
            "TOTAL | {:<4} -> {:<4} | +{:04.2} | +{:05.2}%",
            baseline_chars_per_token,
            total_chars_per_token,
            total_chars_per_token - baseline_chars_per_token,
            ((total_chars_per_token - baseline_chars_per_token) / baseline_chars_per_token * 100.0)
        );
    }

    log::info!("Writing to {:?}.", output);

    tokenizer.save(output).unwrap();
}

fn main() {
    env_logger::Builder::from_default_env().init();

    match flags::Tokengeex::from_env_or_exit().subcommand {
        flags::TokengeexCmd::Train(flags) => {
            train(
                // --- General Purpose ---
                &flags.model,
                &flags.output,
                flags.vocab_size,
                &flags.logfile,
                // --- Data ---
                &flags.train,
                &flags.test,
                // --- Processing ---
                &flags.processor,
                // --- Suggested, Added and Special Tokens ---
                &flags.suggested_tokens_file,
                &flags.added_tokens_file,
                &flags.special_token,
                // --- Initial Vocab ---
                flags.initial_vocab_size.unwrap_or(100000),
                flags.initial_vocab_max_token_length.unwrap_or(24),
                flags.initial_vocab_insert_probability.unwrap_or(0.01),
                &flags.initial_vocab_allow.unwrap_or("^*$".into()),
                // --- Unigram ---
                flags.unigram_shrinking_factor.unwrap_or(0.75),
                flags.unigram_num_sub_iterations.unwrap_or(2),
                &flags.unigram_sample_regularization.unwrap_or("none".into()),
            );
        }
        flags::TokengeexCmd::Evaluate(flags) => {
            evaluate(&flags.vocab, &flags.logfile, &flags.test);
        }
        flags::TokengeexCmd::Encode(flags) => {
            encode(flags.input.as_deref(), &flags.vocab);
        }
        flags::TokengeexCmd::Decode(flags) => {
            decode(flags.input.as_deref(), &flags.vocab);
        }
        flags::TokengeexCmd::Bpe(flags) => {
            bpe(
                &flags.output,
                &flags.vocab,
                flags.num_merges,
                &flags.train,
                flags.step.unwrap_or(10),
                flags.score_scale_factor.unwrap_or(0.75),
                flags.max_merge_length.unwrap_or(16),
                &flags.ignore.unwrap_or("^$".into()),
            );
        }
    }
}
