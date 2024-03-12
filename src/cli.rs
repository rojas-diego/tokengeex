use std::{collections::HashSet, io::BufRead};
use tokengeex::{
    parallelism::MaybeParallelRefIterator, unigram, CapcodeProcessor, CrlfProcessor, Model,
    Processor, ProcessorWrapper, UnicodeProcessor, VocabularyGenerator,
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

fn load_samples<F>(path: &str, process: F) -> Vec<String>
where
    F: Fn(&str) -> String,
{
    let file =
        std::fs::File::open(path).unwrap_or_else(|e| panic!("failed to open {:?}: {:?}", path, e));
    let mut reader = std::io::BufReader::new(file);
    let mut buffer = Vec::new();
    let mut samples = Vec::new();

    loop {
        match reader.read_until(0x00, &mut buffer) {
            Ok(0) => {
                break;
            }
            Ok(_) => {
                let line = String::from_utf8(buffer.clone()).unwrap();
                samples.push(process(line.trim_end_matches('\0')));
                buffer.clear();
            }
            Err(e) => panic!("failed to read from {:?}: {:?}", path, e),
        }
    }

    let samples_total_bytes = samples.iter().map(|s| s.len()).sum::<usize>() as u64;

    log::info!(
        "Loaded {} samples from {:?} ({}).",
        samples.len(),
        path,
        format_bytes_as_mb(samples_total_bytes)
    );

    samples
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

fn load_split(sources: &[String], processor: &[ProcessorWrapper]) -> Vec<(String, Vec<String>)> {
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

            (
                pieces[0].to_string(),
                load_samples(pieces[1], |s| {
                    processor
                        .iter()
                        .fold(s.to_string(), |s, p| p.preprocess(&s))
                }),
            )
        })
        .collect::<Vec<(String, Vec<String>)>>()
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
    vocab_size: usize,
    // --- Data ---
    train: &Vec<String>,
    valid: &Vec<String>,
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
) {
    assert!(
        train.len() > 0,
        "At least one training dataset must be provided"
    );

    let processors = load_processors(processors);
    let added_tokens = load_tokens_files(added_tokens_files, "added");
    let suggested_tokens = load_tokens_files(suggested_tokens_files, "suggested");

    log::info!(
        "Loaded {} added tokens and {} suggested tokens.",
        added_tokens.len(),
        suggested_tokens.len()
    );

    log::info!(
        "Training {:?} model with {} vocabulary entries. Writing to {:?}.",
        model,
        vocab_size,
        output,
    );

    let train_samples = load_split(&train, &processors);
    let valid_samples = load_split(&valid, &processors);
    let test_samples = load_split(&test, &processors);

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

            for (source, samples) in &train_samples {
                log::info!("Collecting frequent tokens from {:?}.", source);

                vocab_generator.feed(&samples);

                log::info!(
                    "Collected frequent tokens from {:?}. Total: {}",
                    source,
                    vocab_generator.current_size()
                );
            }

            let (vocab, keep_indices) =
                vocab_generator.generate(initial_vocab_size, &suggested_tokens, &added_tokens);

            let vocab_total_bytes = vocab.iter().map(|(s, _)| s.len()).sum::<usize>() as u64;

            log::info!(
                "Generated initial vocabulary of size {} ({}).",
                vocab.len(),
                format_bytes_as_mb(vocab_total_bytes)
            );

            let mut model = unigram::Unigram::from(vocab);

            log::info!("Training unigram model.");

            for (source, samples) in &valid_samples {
                log::info!("Evaluating on {:?}.", source);

                let total_tokens = samples
                    .maybe_par_iter()
                    .map(|s| model.encode(s).len())
                    .sum::<usize>();

                let total_bytes = samples.iter().map(|s| s.len()).sum::<usize>();

                log::info!(
                    "Compression on {:?}: {:.2}",
                    source,
                    total_bytes as f64 / total_tokens as f64
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
                flags.vocab_size,
                // --- Data ---
                &flags.train,
                &flags.valid,
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
