use tokengeex::{
    capcode,
    unigram::{
        ScoredToken, SentenceGenerator, Unigram, UnigramTrainerBuilder, VocabularyGenerator,
    },
};

mod flags {
    xflags::xflags! {
        cmd tokengeex {
            /// Train a new tokeniser from data.
            cmd train {
                /// Input dataset filepath. Must be 0x00 separated.
                required -i, --input input: String
                /// Desired vocabulary size.
                required -v, --vocab-size vocab_size: usize
                /// Output vocabulary filepath.
                required -o, --output output: String
                /// Max length of a token in characters.
                optional --max-token-length max_token_length: usize
                /// Max number of words per token.
                optional --max-words-per-token max_words_per_token: usize
                /// Max sentence size.
                optional --max-sentence-size max_sentence_size: usize
                /// How much to shrink the vocabulary at each iteration.
                optional --shrinking-factor shrinking_factor: f64
                /// The size of the initial vocabulary.
                optional --initial-vocab-size initial_vocab_size: usize
                /// Filepath where to cache the initial vocabulary.
                optional --initial-vocab-cache initial_vocab_cache: String
                /// Number of sub-iterations for the EM algorithm.
                optional --num-sub-iterations num_sub_iterations: usize
                /// Suggested token file.
                repeated --suggested-tokens-file suggested_tokens_file: String
                /// Added token file.
                repeated --added-tokens-file added_tokens_file: String
                /// Special token.
                repeated --special-token special_token: String
            }

            /// Encode text using a tokeniser.
            cmd encode {
                /// Input text.
                required -i, --input input: String
                /// Tokeniser vocabulary filepath.
                required -v, --vocab vocab: String
            }

            /// Decode tokenised IDs.
            cmd decode {
                /// Comma separated list of token IDs.
                required -i, --input input: String
                /// Tokeniser vocabulary filepath.
                required -v, --vocab vocab: String
            }

            /// Encode or decode capcode text.
            cmd capcode {
                /// Input text.
                required -i, --input input: String
                /// Decode boolean.
                optional -d, --decode decode: bool
                /// Encode boolean.
                optional -e, --encode encode: bool
            }
        }
    }
}

/// Encode a sample using a tokeniser.
fn encode(input: &str, vocab: &str) {
    let tokenizer = tokengeex::core::load::<Unigram>(vocab).unwrap();

    let encoded = tokenizer.encode(&tokengeex::capcode::encode(input));

    println!("{:?}", encoded);
}

/// Decode a tokenised array of IDs.
fn decode(input: &str, vocab: &str) {
    let tokenizer = tokengeex::core::load::<Unigram>(vocab).unwrap();

    let decoded = tokenizer.decode(
        &input
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect::<Vec<_>>(),
    );

    println!("{}", decoded);
}

/// Encode or decode capcode text.
fn capcode(input: &str, encode: Option<bool>, decode: Option<bool>) {
    let mut encode = encode.unwrap_or(false);
    let decode = decode.unwrap_or(false);

    if encode && decode {
        panic!("Cannot encode and decode at the same time");
    }
    if !encode && !decode {
        encode = true;
    }

    if encode {
        println!("{}", capcode::encode(input));
    } else {
        println!("{}", capcode::decode(input));
    }
}

/// Train a new tokeniser from data.
#[allow(clippy::too_many_arguments)]
fn train(
    input: &str,
    output: &str,
    vocab_size: usize,
    max_token_length: Option<usize>,
    max_words_per_token: Option<usize>,
    max_sentence_size: Option<usize>,
    shrinking_factor: Option<f64>,
    initial_vocab_size: Option<usize>,
    initial_vocab_cache: Option<String>,
    num_sub_iterations: Option<usize>,
    suggested_tokens_files: Vec<String>,
    added_tokens_file: Vec<String>,
    special_tokens: Vec<String>,
) {
    let initial_vocab_size = initial_vocab_size.unwrap_or(vocab_size * 10);
    let max_sentence_size = max_sentence_size.unwrap_or(64);
    let max_words_per_token = max_words_per_token.unwrap_or(2);
    let max_token_length = max_token_length.unwrap_or(24);

    fn collect_tokens(files: Vec<String>) -> Vec<String> {
        files
            .iter()
            .flat_map(|file| {
                let tokens = std::fs::read_to_string(file).unwrap();
                log::info!("Read {} tokens from {:?}", tokens.len(), file);
                serde_json::from_str::<Vec<String>>(&tokens).unwrap()
            })
            .collect()
    }

    let suggested_tokens = collect_tokens(suggested_tokens_files);
    let added_tokens = collect_tokens(added_tokens_file);

    let mut trainer = UnigramTrainerBuilder::default()
        .vocab_size(vocab_size)
        .shrinking_factor(shrinking_factor.unwrap_or(0.1))
        .num_sub_iterations(num_sub_iterations.unwrap_or(4))
        .suggested_tokens(suggested_tokens)
        .added_tokens(added_tokens)
        .special_tokens(special_tokens)
        .build()
        .unwrap();

    log::info!(
        "Training paramaters: vocab_size={}, initial_vocab_size={}, initial_vocab_cache={}, max_token_length={}, max_words_per_token={}, max_sentence_size={}, shrinking_factor={}, num_sub_iterations={}",
        vocab_size, initial_vocab_size, initial_vocab_cache.as_deref().unwrap_or("None"), max_token_length, max_words_per_token, max_sentence_size, trainer.shrinking_factor, trainer.num_sub_iterations);

    let dataset = std::fs::read(input).unwrap();

    log::info!("Read {} bytes from '{}'", dataset.len(), input);

    // The dataset is composed of 0x00 separated samples which are UTF-8
    // encoded. We obtain the samples by splitting the dataset on 0x00 bytes
    // and then converting the resulting byte slices to UTF-8 strings.
    let samples = dataset
        .split(|&b| b == 0x00)
        .map(|s| capcode::encode(String::from_utf8_lossy(s).as_ref()))
        .collect::<Vec<String>>();

    // We can dispose of the dataset to free up memory.
    drop(dataset);

    log::info!("Loaded {} samples", samples.len());

    let sentence_generator = SentenceGenerator::new(max_sentence_size);

    // We feed all samples to the trainer.
    trainer
        .feed(samples.iter(), |sample| {
            Ok(sentence_generator.generate_sentences(sample))
        })
        .unwrap();

    log::info!("Trainer fed");

    let initial_vocab_generator = VocabularyGenerator::new(max_words_per_token, max_token_length);
    let vocab = match initial_vocab_cache {
        Some(initial_vocab_cache) => {
            if let Ok(cache) = std::fs::read_to_string(&initial_vocab_cache) {
                let vocab = serde_json::from_str::<Vec<ScoredToken>>(&cache).unwrap();

                log::info!(
                    "Loaded {} tokens from cached vocab file {:?}",
                    vocab.len(),
                    initial_vocab_cache
                );

                vocab
            } else {
                let str_samples: Vec<&str> = samples.iter().map(AsRef::as_ref).collect();

                let vocab =
                    initial_vocab_generator.generate_vocabulary(&str_samples, initial_vocab_size);

                log::info!(
                    "Generated {} tokens and saved to {:?}",
                    vocab.len(),
                    initial_vocab_cache
                );

                std::fs::write(&initial_vocab_cache, serde_json::to_string(&vocab).unwrap())
                    .unwrap();

                vocab
            }
        }
        None => {
            let str_samples: Vec<&str> = samples.iter().map(AsRef::as_ref).collect();
            let vocab =
                initial_vocab_generator.generate_vocabulary(&str_samples, initial_vocab_size);

            log::info!("Generated {} tokens", vocab.len());

            vocab
        }
    };

    log::info!("Training model");

    let mut model = Unigram::default();
    trainer.train(&mut model, vocab).unwrap();

    let tokenizer = tokengeex::core::Tokenizer::new(model);

    log::info!("Saving model to {:?}", output);

    tokenizer.save(output).unwrap();
}

fn main() {
    env_logger::init();

    match flags::Tokengeex::from_env_or_exit().subcommand {
        flags::TokengeexCmd::Train(flags) => {
            train(
                &flags.input,
                &flags.output,
                flags.vocab_size,
                flags.max_token_length,
                flags.max_words_per_token,
                flags.max_sentence_size,
                flags.shrinking_factor,
                flags.initial_vocab_size,
                flags.initial_vocab_cache,
                flags.num_sub_iterations,
                flags.suggested_tokens_file,
                flags.added_tokens_file,
                flags.special_token,
            );
        }
        flags::TokengeexCmd::Encode(flags) => {
            encode(&flags.input, &flags.vocab);
        }
        flags::TokengeexCmd::Decode(flags) => {
            decode(&flags.input, &flags.vocab);
        }
        flags::TokengeexCmd::Capcode(flags) => {
            capcode(&flags.input, flags.encode, flags.decode);
        }
    }
}
