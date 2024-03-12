use dashmap::DashMap;
use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet},
};
use tokengeex::{
    parallelism::{current_num_threads, MaybeParallelRefIterator, MaybeParallelSlice},
    unigram::ScoredToken,
    CapcodeProcessor, CrlfProcessor, Processor, ProcessorWrapper, UnicodeProcessor,
    VocabularyGenerator,
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

                // --- Data ---
                /// List of source files to train the tokenizer on. Must be
                /// formatted according to {name}:{path}.
                repeated --train input: String
                /// List of source files to validate the tokenizer on. Must be
                /// formatted according to {name}:{path}.
                repeated --valid valid: String
                /// List of source files to test the tokenizer on. Must be
                /// formatted according to {name}:{path}.
                repeated --test test: String

                // --- Processing ---
                /// Apply a processor to the input data.
                repeated --processor processor: String

                // --- Training Options ---
                /// Desired vocabulary size.
                required --vocab-size vocab_size: usize
                /// Maximum token length.
                optional --max-token-length max_token_length: usize

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

                // --- Unigram ---
                /// How much to shrink the vocabulary at each iteration.
                optional --unigram-shrinking-factor unigram_shrinking_factor: f64
                /// Number of sub-iterations for the EM algorithm.
                optional --unigram-num-sub-iterations unigram_num_sub_iterations: usize
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

    let encoded = tokenizer.encode(&input);

    let colors = vec![
        "\x1B[102m", // Bright Green background
        "\x1B[103m", // Bright Yellow background
        "\x1B[104m", // Bright Blue background
    ];

    let encoded = encoded
        .iter()
        .map(|id| tokenizer.id_to_token(*id).unwrap().to_string())
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

    let decoded = tokenizer.decode(
        &input
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect::<Vec<_>>(),
    );

    println!("{}", decoded);
}

fn mmap_files(files: &Vec<String>) -> Vec<memmap2::Mmap> {
    files
        .iter()
        .map(|filename| unsafe {
            let file = std::fs::File::open(filename);

            if file.is_err() {
                panic!("Could not open file {:?}", filename);
            }

            let mmap = memmap2::Mmap::map(&file.unwrap()).unwrap();

            log::info!(
                "Successfuly mapped {:?} ({}).",
                filename,
                format_bytes_as_mb(mmap.len() as u64)
            );

            mmap
        })
        .collect::<Vec<_>>()
}

fn split_mmaps<'a>(mmaps: &'a Vec<memmap2::Mmap>, files: &'a Vec<String>) -> Vec<Vec<&'a str>> {
    mmaps
        .iter()
        .zip(files.iter())
        .map(|(mmap, file)| {
            let samples: Vec<&str> = mmap
                .split(|&x| x == b'\0')
                .map(|slice| {
                    std::str::from_utf8(slice)
                        .expect("train, valid and test samples must be valid UTF-8")
                })
                .filter(|s| !s.is_empty())
                .collect();

            log::info!("Loaded {} samples from {:?}", samples.len(), file);

            samples
        })
        .collect()
}

fn format_bytes_as_mb(bytes: u64) -> String {
    format!("{:.2}MB", bytes as f64 / 1_000_000.0)
}

/// Train a new tokeniser from data.
#[allow(clippy::too_many_arguments)]
fn train(
    // --- General Purpose ---
    model: &str,
    output: &str,
    // --- Data ---
    train: &Vec<String>,
    valid: &Vec<String>,
    test: &Vec<String>,
    // --- Processing ---
    processor: &Vec<String>,
    // --- Training Options ---
    vocab_size: usize,
    max_token_length: usize,
    // --- Suggested, Added and Special Tokens ---
    suggested_tokens_files: &Vec<String>,
    added_tokens_files: &Vec<String>,
    special_tokens: &Vec<String>,
    // --- Initial Vocabulary ---
    initial_vocab_size: usize,
    initial_vocab_insert_probability: f64,
    initial_vocab_allow: &str,
    // --- Unigram ---
    unigram_shrinking_factor: f64,
    unigram_num_sub_iterations: usize,
) {
    assert!(
        train.len() > 0,
        "At least one training dataset must be provided"
    );

    let processors = processor
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
        .collect::<Vec<_>>();

    let added_tokens: HashSet<String> = added_tokens_files
        .iter()
        .flat_map(|filename| {
            let file = std::fs::read_to_string(filename)
                .unwrap_or_else(|_| panic!("Could not read added tokens file {:?}", filename));

            let tokens: Vec<String> = serde_json::from_str(&file)
                .unwrap_or_else(|_| panic!("Could not parse added tokens file {:?}", filename));

            log::info!("Loaded {} added tokens from {:?}", tokens.len(), filename);

            tokens
        })
        .collect();

    let suggested_tokens: HashSet<String> = suggested_tokens_files
        .iter()
        .flat_map(|filename| {
            let file = std::fs::read_to_string(filename)
                .unwrap_or_else(|_| panic!("Could not read suggested tokens file {:?}", filename));

            let tokens: Vec<String> = serde_json::from_str(&file)
                .unwrap_or_else(|_| panic!("Could not parse suggested tokens file {:?}", filename));

            log::info!(
                "Loaded {} suggested tokens from {:?}",
                tokens.len(),
                filename
            );

            tokens
        })
        .collect();

    log::info!(
        "Loaded {} added tokens and {} suggested tokens.",
        added_tokens.len(),
        suggested_tokens.len()
    );

    // We mmap the training, validation and test datasets to avoid loading
    // them into memory.
    let train_mmaps = mmap_files(train);
    let valid_mmaps = mmap_files(valid);
    let test_mmaps = mmap_files(test);

    // We expect each mmaped file to be an array of 0x00 separated UTF-8
    // strings.
    let train_samples = split_mmaps(&train_mmaps, &train);
    let valid_samples = split_mmaps(&valid_mmaps, &valid);
    let test_samples = split_mmaps(&test_mmaps, &test);

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
                initial_vocab_size, max_token_length, initial_vocab_insert_probability
            );

            log::info!("Using allow rule: {:?}", initial_vocab_allow);

            let vocabulary_generator = VocabularyGenerator::new(
                max_token_length,
                initial_vocab_insert_probability,
                initial_vocab_allow,
            );

            let frequent_tokens: DashMap<String, usize> = DashMap::new();

            for (samples, source) in train_samples.iter().zip(train.iter()) {
                log::info!("Preprocessing {:?} dataset", source);

                let preprocessed_samples = samples
                    .maybe_par_iter()
                    .map(|s| {
                        processors
                            .iter()
                            .fold(s.to_string(), |s, p| p.preprocess(&s))
                    })
                    .collect::<Vec<String>>();

                log::info!(
                    "Preprocessed {} samples from {:?}. Collecting frequent tokens.",
                    preprocessed_samples.len(),
                    source
                );

                let chunk_size =
                    std::cmp::max(preprocessed_samples.len() / current_num_threads(), 1);

                let num_tokens = preprocessed_samples
                    .maybe_par_chunks(chunk_size)
                    .map(|chunk| {
                        let tokens = vocabulary_generator
                            .collect_frequent_tokens(chunk.iter().map(|s| s.as_str()));

                        for (token, count) in &tokens {
                            *frequent_tokens.entry(token.to_string()).or_insert(0) += count;
                        }

                        tokens.len()
                    })
                    .reduce(|| 0, |a, b| a + b);

                log::info!(
                    "Collected {} frequent tokens from {:?}. Total: {}",
                    num_tokens,
                    source,
                    frequent_tokens.len(),
                );
            }

            let frequent_tokens = frequent_tokens
                .into_iter()
                .collect::<HashMap<String, usize>>();

            // Collect the frequency of the suggested tokens.
            // TODO: Instead of relying on a default frequency of 1, we should
            // consider the frequency of the suggested tokens in the dataset. This
            // involves making a second pass over the dataset.
            let suggested_tokens_freq = suggested_tokens
                .iter()
                .map(|token| frequent_tokens.get(token.as_str()).copied().unwrap_or(1))
                .collect::<Vec<usize>>();

            // Convert the tokens to a vector and sort them by frequency.
            log::info!("Sorting {} frequent tokens.", frequent_tokens.len());
            let mut frequent_tokens: Vec<_> = frequent_tokens.into_iter().collect();
            frequent_tokens.sort_by_key(|(_, freq)| Reverse(*freq));

            // Keep track of duplicates, ensuring the earlier occurrence is kept.
            let mut seen: HashSet<&str> = HashSet::new();

            // Add all 256 ASCII characters and byte values to the initial
            // vocabulary. We assume the frequency of each byte is the same as
            // the highest frequency token.
            let highest_freq = frequent_tokens.first().map(|(_, freq)| *freq).unwrap_or(1);
            let mut vocab: Vec<ScoredToken> = (0..255_u8)
                .map(|b| (vec![b], highest_freq as f64))
                .collect();

            // Add the suggested tokens.
            for (i, token) in suggested_tokens.iter().enumerate() {
                if !seen.contains(token.as_str()) {
                    seen.insert(token);
                    vocab.push((
                        token.as_bytes().to_vec(),
                        (suggested_tokens_freq[i] as f64) * (token.len() as f64),
                    ));
                }
            }

            // We further add the most frequent substrings.
            for (token, freq) in &frequent_tokens {
                if vocab.len() >= vocab_size {
                    break;
                }

                if !seen.contains(token.as_str()) {
                    seen.insert(token.as_str());
                    vocab.push((token.as_bytes().to_vec(), (freq * token.len()) as f64));
                }
            }

            // Sort the vocabulary by score.
            vocab.sort_by(|(_, a), (_, b)| {
                a.partial_cmp(b)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .reverse()
            });

            // Convert the scores to log probabilities.
            logprobs(&mut vocab);

            // Computing log probabilities generates NaNs for items where freq=0.
            vocab.iter_mut().for_each(|(_, score)| {
                if !score.is_normal() {
                    *score = 0.0;
                }
            });
        }
        _ => {
            panic!("Model {:?} is not supported.", model);
        }
    }
}

fn logprobs(pieces: &mut [ScoredToken]) {
    let sum: f64 = pieces.iter().map(|(_, score)| score).sum();
    let logsum = sum.ln();
    for (_, score) in pieces.iter_mut() {
        *score = score.ln() - logsum;
    }
}

fn main() {
    env_logger::init();

    match flags::Tokengeex::from_env_or_exit().subcommand {
        flags::TokengeexCmd::Train(flags) => {
            train(
                // --- General Purpose ---
                &flags.model,
                &flags.output,
                // --- Data ---
                &flags.train,
                &flags.valid,
                &flags.test,
                // --- Processing ---
                &flags.processor,
                // --- Training Options ---
                flags.vocab_size,
                flags.max_token_length.unwrap_or(24),
                // --- Suggested, Added and Special Tokens ---
                &flags.suggested_tokens_file,
                &flags.added_tokens_file,
                &flags.special_token,
                // --- Initial Vocab ---
                flags.initial_vocab_size.unwrap_or(100000),
                flags.initial_vocab_insert_probability.unwrap_or(0.01),
                &flags.initial_vocab_allow.unwrap_or("^*$".into()),
                // --- Unigram ---
                flags.unigram_shrinking_factor.unwrap_or(0.75),
                flags.unigram_num_sub_iterations.unwrap_or(2),
            );
        }
        flags::TokengeexCmd::Encode(flags) => {
            encode(flags.input.as_deref(), &flags.vocab);
        }
        flags::TokengeexCmd::Decode(flags) => {
            decode(flags.input.as_deref(), &flags.vocab);
        }
    }
}
