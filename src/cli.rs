use tokengeex::{
    capcode,
    unigram::{ScoredToken, Unigram, UnigramTrainerBuilder, Vocab, VocabularyGenerator},
};

mod flags {
    xflags::xflags! {
        cmd tokengeex {
            /// Train a new tokeniser from data.
            cmd train {
                // --- General Purpose ---
                /// Kind of model to train. Unigram is the only one supported.
                required -m, --model model: String
                /// Input dataset filepath. Must be 0x00 separated.
                required -i, --input input: String
                /// Output vocabulary filepath.
                required -o, --output output: String

                // --- Tokenizer ---
                /// Special token.
                repeated --special-token special_token: String

                // --- Model Trainer Parameters ---
                /// Desired vocabulary size.
                required -v, --vocab-size vocab_size: usize
                /// How much to shrink the vocabulary at each iteration.
                optional --shrinking-factor shrinking_factor: f64
                /// Number of sub-iterations for the EM algorithm.
                optional --num-sub-iterations num_sub_iterations: usize
                /// Suggested token file.
                repeated --suggested-tokens-file suggested_tokens_file: String
                /// Added token file.
                repeated --added-tokens-file added_tokens_file: String

                // --- Vocabulary Generator ---
                /// Max length of a token in characters.
                optional --vg-max-token-length vg_max_token_length: usize
                /// Max number of words per token.
                optional --vg-max-words-per-token vg_max_words_per_token: usize
                /// The size of the initial vocabulary.
                optional --vg-initial-vocab-size vg_initial_vocab_size: usize
                /// Probability of inserting a new token to the vocabulary.
                optional --vg-insert-probability vg_insert_probability: f64
                /// Filepath where to cache the initial vocabulary.
                optional --vg-cache vg_cache: String
                /// Strict boolean.
                optional --vg-strict vg_strict: bool
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

    if encode {
        println!("{}", capcode::encode(&input));
    } else {
        println!("{}", capcode::decode(&input));
    }
}

/// Train a new tokeniser from data.
#[allow(clippy::too_many_arguments)]
fn train(
    // --- General Purpose ---
    model: &str,
    input: &str,
    output: &str,
    // --- Tokenizer ---
    special_tokens: Vec<String>,
    // --- Model Trainer Parameters ---
    vocab_size: usize,
    shrinking_factor: f64,
    num_sub_iterations: usize,
    suggested_tokens_files: Vec<String>,
    added_tokens_files: Vec<String>,
    // --- Vocabulary Generator ---
    vg_max_token_length: usize,
    vg_max_words_per_token: usize,
    vg_initial_vocab_size: usize,
    vg_insert_probability: f64,
    vg_cache: Option<String>,
    vg_strict: bool,
) {
    assert!(model == "unigram", "Only 'unigram' model is supported");

    log::info!("Training {} model on '{}'", model, input);
    log::info!("Special tokens: {:?}", special_tokens);
    log::info!(
        "Model Trainer Parameters: vocab_size={}, shrinking_factor={}, num_sub_iterations={}",
        vocab_size,
        shrinking_factor,
        num_sub_iterations
    );
    log::info!(
        "Vocabulary Generator: max_token_length={}, max_words_per_token={}, initial_vocab_size={}, insert_probability={}, cache={} strict={}",
        vg_max_token_length,
        vg_max_words_per_token,
        vg_initial_vocab_size,
        vg_insert_probability,
        vg_cache.as_deref().unwrap_or("None"),
        vg_strict
    );

    fn collect_tokens(files: Vec<String>) -> Vec<String> {
        files
            .iter()
            .flat_map(|file| {
                let tokens =
                    serde_json::from_str::<Vec<String>>(&std::fs::read_to_string(file).unwrap())
                        .unwrap();
                log::info!("Read {} tokens from {:?}", tokens.len(), file);
                tokens
            })
            .collect()
    }

    let suggested_tokens = collect_tokens(suggested_tokens_files);
    let added_tokens = collect_tokens(added_tokens_files);

    let mut trainer = UnigramTrainerBuilder::default()
        .vocab_size(vocab_size)
        .shrinking_factor(shrinking_factor)
        .num_sub_iterations(num_sub_iterations)
        .added_tokens(added_tokens)
        .build()
        .unwrap();

    let dataset = std::fs::read(input).unwrap();

    log::info!("Read {} bytes from '{}'", dataset.len(), input);

    // The dataset is composed of 0x00 separated samples which are UTF-8
    // encoded. We obtain the samples by splitting the dataset on 0x00 bytes
    // and then converting the resulting byte slices to UTF-8 strings.
    let samples: Vec<String> = dataset
        .split(|&b| b == 0x00)
        .map(|s| tokengeex::capcode::encode(&String::from_utf8_lossy(s)))
        .collect();

    // We can dispose of the dataset to free up memory.
    drop(dataset);

    samples.iter().for_each(|s| trainer.feed(s));

    log::info!("Loaded {} samples", samples.len());

    let initial_vocab_generator = VocabularyGenerator::new(
        vg_max_words_per_token,
        vg_max_token_length,
        vg_insert_probability,
        suggested_tokens,
        vg_strict,
    );
    let vocab = match vg_cache {
        Some(vg_cache) => {
            if let Ok(cache) = std::fs::read_to_string(&vg_cache) {
                let vocab: Vec<ScoredToken> = serde_json::from_str::<Vocab>(&cache).unwrap().into();

                log::info!(
                    "Loaded {} tokens from cached vocab file {:?}",
                    vocab.len(),
                    vg_cache
                );

                vocab
            } else {
                let vocab = initial_vocab_generator
                    .generate_vocabulary(samples.iter().map(AsRef::as_ref), vg_initial_vocab_size);

                log::info!(
                    "Generated {} tokens and saved to {:?}",
                    vocab.len(),
                    vg_cache
                );

                std::fs::write(
                    &vg_cache,
                    serde_json::to_string(&Vocab::from(vocab.clone())).unwrap(),
                )
                .unwrap();

                vocab
            }
        }
        None => {
            let vocab = initial_vocab_generator
                .generate_vocabulary(samples.iter().map(AsRef::as_ref), vg_initial_vocab_size);

            log::info!("Generated {} tokens", vocab.len());

            vocab
        }
    };

    log::info!("Training model");

    let mut model = Unigram::default();
    trainer.train(&mut model, vocab).unwrap();

    let mut tokenizer = tokengeex::core::Tokenizer::from(model);

    let special_tokens: Vec<&str> = special_tokens.iter().map(AsRef::as_ref).collect();

    tokenizer.add_special_tokens(special_tokens.as_slice());

    log::info!("Saving model to {:?}", output);

    tokenizer.save(output).unwrap();
}

fn main() {
    env_logger::init();

    match flags::Tokengeex::from_env_or_exit().subcommand {
        flags::TokengeexCmd::Train(flags) => {
            train(
                // --- General Purpose ---
                &flags.model,
                &flags.input,
                &flags.output,
                // --- Tokenizer ---
                flags.special_token,
                // --- Model Trainer Parameters ---
                flags.vocab_size,
                flags.shrinking_factor.unwrap_or(0.75),
                flags.num_sub_iterations.unwrap_or(4),
                flags.suggested_tokens_file,
                flags.added_tokens_file,
                // --- Vocabulary Generator ---
                flags.vg_max_token_length.unwrap_or(24),
                flags.vg_max_words_per_token.unwrap_or(3),
                flags.vg_initial_vocab_size.unwrap_or(100000),
                flags.vg_insert_probability.unwrap_or(0.02),
                flags.vg_cache,
                flags.vg_strict.unwrap_or(false),
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
