use std::{collections::HashMap, ops::Deref};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tokengeex::{
    CapcodeProcessor, CrlfProcessor, Processor, ProcessorWrapper, UnicodeProcessor,
    VocabularyGenerator,
};
use unicode_normalization::UnicodeNormalization;

mod flags {
    xflags::xflags! {
        cmd tokengeex {
            /// Train a new tokeniser from data.
            cmd train {
                // --- General Purpose ---
                /// Kind of model to train. Unigram is the only one supported.
                required -m, --model model: String
                /// Output vocabulary filepath.
                required -o, --output output: String

                // --- Data ---
                /// Dataset to train the tokenizer on.
                repeated --train input: String
                /// Dataset to validate the tokenizer on.
                repeated --valid valid: String
                /// Dataset to test the tokenizer on.
                repeated --test test: String

                // --- Processing ---
                /// Apply a processor to the input data.
                repeated --processor processor: String

                // --- Training Options ---
                /// Desired vocabulary size.
                required -v, --vocab-size vocab_size: usize
                /// Maximum token length.
                optional --max-token-length max_token_length: usize
                /// Substrings that match allow Regexes will be considered for
                /// the vocabulary.
                repeated --allow allow: String
                /// Substrings that match disallow Regexes will not be
                /// considered for the vocabulary.
                repeated --disallow disallow: String

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
                /// Filepath where to cache the initial vocabulary.
                optional --initial-vocab-cache initial_vocab_cache: String

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

            /// Encode or decode capcode text.
            cmd capcode {
                /// Input text. Otherwise, stdin is used.
                optional -i, --input input: String
                /// Decode boolean.
                optional -d, --decode decode: bool
                /// Encode boolean.
                optional -e, --encode encode: bool
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

    let tokenizer = tokengeex::core::load(vocab).unwrap();

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

    let tokenizer = tokengeex::core::load(vocab).unwrap();

    let decoded = tokenizer.decode(
        &input
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect::<Vec<_>>(),
    );

    println!("{}", decoded);
}

/// Encode or decode capcode text.
fn capcode(input: Option<&str>, encode: Option<bool>, decode: Option<bool>) {
    let input = input.map_or_else(
        || std::io::read_to_string(&mut std::io::stdin()).unwrap(),
        |s| s.to_string(),
    );

    let mut encode = encode.unwrap_or(false);
    let decode = decode.unwrap_or(false);

    if encode && decode {
        panic!("Cannot encode and decode at the same time");
    }
    if !encode && !decode {
        encode = true;
    }

    let processor = CapcodeProcessor;

    if encode {
        println!("{}", processor.preprocess(&input));
    } else {
        println!("{}", processor.postprocess(&input));
    }
}

fn mmap_files(files: &Vec<String>) -> Vec<memmap2::Mmap> {
    files
        .iter()
        .map(|file| unsafe {
            let mmap = memmap2::Mmap::map(&std::fs::File::open(file).unwrap()).unwrap();
            log::info!(
                "Successfuly mapped {:?} ({}).",
                file,
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
    allow: &Vec<String>,
    disallow: &Vec<String>,
    // --- Suggested, Added and Special Tokens ---
    suggested_tokens_files: &Vec<String>,
    added_tokens_files: &Vec<String>,
    special_tokens: &Vec<String>,
    // --- Initial Vocabulary ---
    initial_vocab_size: usize,
    initial_vocab_insert_probability: f64,
    initial_vocab_cache: Option<&str>,
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

            for (i, disallow) in disallow.iter().enumerate() {
                log::info!("Using disallow rule #{}: {:?}", i, disallow);
            }

            let vocabulary_generator = VocabularyGenerator::new(
                max_token_length,
                initial_vocab_insert_probability,
                allow,
                disallow,
            );

            for (samples, source) in train_samples.iter().zip(train.iter()) {
                log::info!("Preprocessing {:?} dataset", source);

                let preprocessed_samples = samples
                    .par_iter()
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

                let frequent_tokens = vocabulary_generator
                    .collect_frequent_tokens(preprocessed_samples.iter().map(|s| s.as_str()));

                log::info!(
                    "Collected {} frequent tokens from training dataset.",
                    frequent_tokens.len()
                );
            }
        }
        _ => {
            panic!("Model {:?} is not supported.", model);
        }
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
                &flags.allow,
                &flags.disallow,
                // --- Suggested, Added and Special Tokens ---
                &flags.suggested_tokens_file,
                &flags.added_tokens_file,
                &flags.special_token,
                // --- Initial Vocab ---
                flags.initial_vocab_size.unwrap_or(100000),
                flags.initial_vocab_insert_probability.unwrap_or(0.01),
                flags.initial_vocab_cache.as_deref(),
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
        flags::TokengeexCmd::Capcode(flags) => {
            capcode(flags.input.as_deref(), flags.encode, flags.decode);
        }
    }
}
