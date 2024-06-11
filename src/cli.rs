use ::regex::Regex;
use fancy_regex::Regex as FancyRegex;
use rand::prelude::SliceRandom;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tokengeex::{CrlfProcessor, Model, Processor, ProcessorWrapper, Tokenizer, UnicodeProcessor};

mod filter;
mod generate;
mod merge;
mod mine;
mod prune;
mod regex;

pub use filter::*;
pub use generate::*;
pub use merge::*;
pub use mine::*;
pub use prune::*;
pub use regex::*;

mod flags {
    xflags::xflags! {
        cmd tokengeex {
            /// Create a new tokenizer with a vocabulary generated from a large
            /// training dataset.
            cmd generate {
                /// The size of the vocabulary to generate.
                required -v, --vocab-size vocab_size: usize
                /// The output file to save the tokenizer.
                required -o, --output output: String

                // --- Processing ---
                /// Apply a processor to the input data.
                repeated --processor processor: String

                // --- Data ---
                /// List of source files to train the tokenizer on. Must be
                /// formatted according to {name}:{path}[:proportion].
                repeated --train input: String

                // --- Suggested, Added and Special Tokens ---
                /// Special token.
                repeated --special special: String
                /// Path to a file which contains an array of suggested tokens.
                repeated --suggested suggested: String
                /// Path to a file which contains an array of added tokens.
                repeated --added added: String

                // --- Options ---
                /// Path to a file which contains a regular expression. Only
                /// tokens that match this regex will be considered.
                optional --allow allow: String
                /// Path to a file which contains a regular expression. If
                /// specified, every sample will be split according to this
                /// regex before being processed. Supports fancy regex syntax.
                optional --split split: String
                /// Probability of inserting a new token to the vocabulary.
                optional --insert-probability insert_probability: f64
                /// Maximum token length.
                optional --max-token-length max_token_length: usize
            }

            /// Iteratively prune the vocabulary by removing the least frequent
            /// tokens.
            cmd prune {
                /// The input tokenizer file.
                required -i, --input input: String
                /// The output tokenizer file.
                required -o, --output output: String
                /// The size of the vocabulary to prune to.
                required -v, --vocab-size vocab_size: usize

                // --- Data ---
                /// List of source files to train the tokenizer on. Must be
                /// formatted according to {name}:{path}[:proportion].
                repeated --train input: String

                // --- Options ---
                /// Dropout factor. The probability of omitting a token from
                /// the segmentation of a sample.
                optional --dropout dropout: f64
                /// How much to shrink the vocabulary at each iteration.
                optional --shrink-factor shrink_factor: f64
                /// Number of sub-iterations for the EM algorithm.
                optional --em-subiters em_subiters: usize
            }

            /// Filter the vocabulary by removing entries that match a given
            /// regex or condition.
            cmd filter {
                /// The input tokenizer file.
                required -i, --input input: String
                /// The output tokenizer file.
                required -o, --output output: String
                /// Do not filter past this vocabulary size. Default is 0.
                optional -v, --vocab-size vocab_size: usize

                // --- Options ---
                /// Filters tokens with a log probability lower than this value.
                optional --min-score min_score: f64
                /// Force. Removes "keep" tokens if they match the filter.
                optional --force force: bool
            }

            /// Merge the most frequent pairs of tokens in the vocabulary.
            cmd merge {
                /// The input tokenizer file.
                required -i, --input input: String
                /// The output tokenizer file.
                required -o, --output output: String

                // --- Data ---
                /// List of source files to train the tokenizer on. Must be
                /// formatted according to {name}:{path}[:proportion].
                repeated --train input: String

                // --- Options ---
                /// Path to a file which contains a regular expression. Only
                /// merges that match this regex will be considered.
                required --allow allow: String
                /// The number of merges to perform.
                optional --num-merges num_merges: usize
                /// How many new tokens to merge at each iteration.
                optional --step step: usize
                /// The score of each new token will be caculated based on
                /// the sum of the scores of the tokens that were merged times
                /// this factor.
                optional --scale-factor scale_factor: f64
                /// Maximum size of a token.
                optional --max-token-length max_token_length: usize
            }

            /// Generate a Regex for downstream use with TokenGeeX.
            cmd regex {
                /// Output file to save the Regex.
                optional -o, --output output: String
                /// Pattern to include. Can be either a named regex or a
                /// custom regex.
                repeated -p, --pattern pattern: String
            }

            /// Mine for common idioms from a large scale dataset.
            cmd mine {
                /// Number of idioms to keep from the set of most occuring
                /// idioms.
                required -n, --num-idioms num_idioms: usize
                /// Output file to save the idioms.
                required -o, --output output: String

                // --- Data ---
                /// List of source files to train the tokenizer on. Must be
                /// formatted according to {name}:{path}[:proportion].
                repeated --train input: String

                // --- Options ---
                /// Pattern to look for. Can be either a named regex or a
                /// custom regex.
                repeated -p, --pattern pattern: String
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

pub enum Error {
    NoPath(usize, usize),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::NoPath(pos, len) => write!(f, "no path to position {}/{}", pos, len),
        }
    }
}

impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::NoPath(pos, len) => write!(f, "NoPath({}, {})", pos, len),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub struct Source {
    pub name: String,
    pub processed_samples: Vec<String>,
    #[allow(dead_code)]
    pub total_bytes: usize,
    #[allow(dead_code)]
    pub processed_total_bytes: usize,
    #[allow(dead_code)]
    pub total_chars: usize,
    #[allow(dead_code)]
    pub processed_total_chars: usize,
}

fn load_processors(processors: &[String]) -> Vec<ProcessorWrapper> {
    processors
        .iter()
        .map(|name| {
            let processor = match name.as_str() {
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

fn load_sources(sources: &[String], processors: &[ProcessorWrapper], mode: &str) -> Vec<Source> {
    sources
        .par_iter()
        .map(|source| {
            let pieces = source.split(':').collect::<Vec<&str>>();
            if pieces.len() < 2 || pieces.len() > 3 {
                panic!(
                    "Invalid source format: {:?}. Expected to be formatted as {{name}}:{{path}}",
                    source
                );
            }
            let name = pieces[0];
            let filepath = pieces[1];
            let proportion = pieces
                .get(2)
                .map(|s| {
                    s.parse::<f64>().unwrap_or_else(|_| {
                        panic!("Invalid proportion {:?} in source {:?}", s, source);
                    })
                })
                .unwrap_or(1.0);

            let file_contents = std::fs::read(filepath).unwrap_or_else(|e| {
                panic!("Failed to open/read {:?}: {:?}", filepath, e);
            });

            let samples: Vec<&str> = file_contents
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
                .take((samples.len() as f64 * proportion) as usize)
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
            let processed_total_chars = processed_samples
                .iter()
                .map(|s| s.chars().count())
                .sum::<usize>();

            log::info!(
                "Loaded {}/{} samples from {:?} {} source ({})",
                processed_samples.len(),
                samples.len(),
                name,
                mode,
                format_bytes_as_mb(processed_total_bytes as u64),
            );

            Source {
                name: name.to_string(),
                processed_samples,
                total_bytes,
                processed_total_bytes,
                total_chars,
                processed_total_chars,
            }
        })
        .collect()
}

fn load_regex(path: &str) -> Regex {
    Regex::new(
        std::fs::read_to_string(path)
            .unwrap()
            .replace(['\n', '\r'], "")
            .trim(),
    )
    .unwrap()
}

fn load_fancy_regex(path: &str) -> FancyRegex {
    FancyRegex::new(
        std::fs::read_to_string(path)
            .unwrap()
            .replace(['\n', '\r'], "")
            .trim(),
    )
    .unwrap()
}

fn load_patterns(patterns: &[String]) -> Vec<Regex> {
    patterns
        .iter()
        .map(|name| {
            PATTERNS
                .iter()
                .find(|(n, _, _, _)| n == name)
                .map(|(_, pattern, _, _)| pattern())
                .unwrap_or_else(|| {
                    Regex::new(name).unwrap_or_else(|e| {
                        panic!("Failed to parse pattern {:?} as a regex: {:?}", name, e)
                    })
                })
        })
        .collect::<Vec<Regex>>()
}

fn load_tokens(tokens: &[String], mode: &str) -> Vec<String> {
    tokens
        .iter()
        .flat_map(|path| {
            let tokens: Vec<String> = serde_json::from_str(
                &std::fs::read_to_string(path)
                    .unwrap_or_else(|_| panic!("Failed to read tokens from {:?}", path)),
            )
            .unwrap();

            log::info!("Loaded {} {} tokens from {:?}", tokens.len(), mode, path);

            tokens
        })
        .collect()
}

fn shuffled_train_samples(sources: &[Source]) -> Vec<&str> {
    let mut train_samples = sources
        .iter()
        .flat_map(|source| source.processed_samples.iter())
        .map(|s| s.as_str())
        .collect::<Vec<&str>>();
    let mut rng = rand::thread_rng();
    train_samples.shuffle(&mut rng);
    train_samples
}

fn format_bytes_as_mb(bytes: u64) -> String {
    format!("{:.2}MB", bytes as f64 / 1_000_000.0)
}

#[allow(clippy::too_many_arguments)]
fn generate_cmd(
    output: &str,
    vocab_size: usize,
    sources: &[String],
    processors: &[String],
    special_tokens: &[String],
    suggested_tokens: &[String],
    added_tokens: &[String],
    split: Option<String>,
    allow: Option<String>,
    insert_probability: f64,
    max_token_length: usize,
) {
    log::info!(
        "Generating vocabulary output={:?} vocab_size={} split={:?} allow={:?} insert_probability={} max_token_length={}",
        output,
        vocab_size,
        split,
        allow,
        insert_probability,
        max_token_length
    );

    let processors = load_processors(processors);
    let train = load_sources(sources, &processors, "train");
    let allow_regex = allow.map(|allow| load_regex(&allow));
    let split_regex = split.map(|split| load_fancy_regex(&split));
    let added_tokens = load_tokens(added_tokens, "added");
    let suggested_tokens = load_tokens(suggested_tokens, "suggested");

    log::debug!("Allow regex: {:?}", allow_regex);
    log::debug!("Split regex: {:?}", split_regex);

    let mut vocab_generator = VocabularyGenerator::new(
        max_token_length,
        insert_probability,
        split_regex,
        allow_regex,
        added_tokens,
        suggested_tokens,
    );

    for source in &train {
        vocab_generator.feed(&source.processed_samples);

        log::info!(
            "Collected frequent tokens from {:?}. Total: {}",
            source.name,
            vocab_generator.current_size()
        );
    }

    let vocab = vocab_generator.generate(vocab_size);

    log::info!(
        "Generated initial vocabulary vocab_size={} mem={}",
        vocab.len(),
        format_bytes_as_mb(vocab.iter().map(|token| token.len()).sum::<usize>() as u64)
    );

    let model = Model::from(vocab);
    let tokenizer = Tokenizer::new(model, processors, special_tokens);

    tokenizer.save(output).unwrap();

    log::info!("Saved vocabulary to {:?}", output);
}

#[allow(clippy::too_many_arguments)]
fn prune_cmd(
    input: &str,
    output: &str,
    vocab_size: usize,
    train: &[String],
    dropout: f64,
    shrink_factor: f64,
    em_subiters: usize,
) {
    log::info!(
        "Pruning vocabulary input={:?} output={:?} vocab_size={} dropout={} shrink_factor={} em_subiters={}",
        input,
        output,
        vocab_size,
        dropout,
        shrink_factor,
        em_subiters
    );

    let (mut model, processors, special_tokens) = Tokenizer::from_file(input).unwrap().into_inner();
    let initial_vocab_size = model.vocab_size();
    let train = load_sources(train, &processors, "train");
    let train_samples = shuffled_train_samples(&train);

    let vocab_pruner = ModelVocabularyPruner::new(vocab_size, shrink_factor, em_subiters, dropout);

    vocab_pruner.prune(&mut model, &train_samples).unwrap();

    log::info!(
        "Pruned vocabulary from={} to={} mem={}",
        initial_vocab_size,
        vocab_size,
        format_bytes_as_mb(model.vocab().iter().map(|token| token.len()).sum::<usize>() as u64)
    );

    let tokenizer = Tokenizer::new(model, processors, special_tokens);
    tokenizer.save(output).unwrap();

    log::info!("Saved pruned vocabulary to {:?}", output);
}

#[allow(clippy::too_many_arguments)]
fn filter_cmd(input: &str, output: &str, vocab_size: usize, min_score: Option<f64>, force: bool) {
    log::info!(
        "Filtering vocabulary input={:?} output={:?} vocab_size={} min_score={:?} force={}",
        input,
        output,
        vocab_size,
        min_score,
        force
    );

    let (mut model, processors, special_tokens) = Tokenizer::from_file(input).unwrap().into_inner();
    let initial_vocab_size = model.vocab_size();

    let vocab_filter = VocabularyFilter::new(vocab_size, min_score, force);
    vocab_filter.filter(&mut model);

    log::debug!(
        "Filtered vocabulary from={} to={} mem={}",
        initial_vocab_size,
        model.vocab_size(),
        format_bytes_as_mb(model.vocab().iter().map(|token| token.len()).sum::<usize>() as u64)
    );

    let tokenizer = Tokenizer::new(model, processors, special_tokens);
    tokenizer.save(output).unwrap();

    log::info!("Saved filtered vocabulary to {:?}", output);
}

#[allow(clippy::too_many_arguments)]
fn regex_cmd(output: &Option<String>, patterns: &[String]) {
    match output {
        None => {
            for (name, pattern, _, _) in PATTERNS {
                println!("{}: {}", name, pattern());
            }
        }
        Some(output) => {
            log::info!(
                "Generating regex output={:?} patterns={:?}",
                output,
                patterns.len(),
            );

            let patterns = load_patterns(patterns);
            let re = build_allow_regex(patterns);

            log::debug!("Generated regex: {:?}", re);

            std::fs::write(output, re.as_str()).unwrap();

            log::info!("Saved regex to {:?}", output);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn merge_cmd(
    input: &str,
    output: &str,
    train: &[String],
    allow: &str,
    num_merges: usize,
    step: usize,
    scale_factor: f64,
    max_token_length: usize,
) {
    assert!(
        !train.is_empty(),
        "At least one train source must be provided."
    );

    log::info!(
        "Merging vocabulary input={:?} output={:?} num_merges={} step={} scale_factor={} max_token_length={}",
        input,
        output,
        num_merges,
        step,
        scale_factor,
        max_token_length
    );

    let (mut model, processors, special_tokens) = Tokenizer::from_file(input).unwrap().into_inner();
    let train = load_sources(train, &processors, "train");
    let train_samples = shuffled_train_samples(&train);
    let initial_vocab_size = model.vocab_size();
    let allow_regex = load_regex(allow);

    let vocab_merger = ModelVocabularyMerger::new(
        allow_regex,
        num_merges,
        step,
        scale_factor,
        max_token_length,
    );

    vocab_merger.merge(&mut model, &train_samples);

    log::info!(
        "Merged vocabulary from={} to={} mem={}",
        initial_vocab_size,
        model.vocab_size(),
        format_bytes_as_mb(model.vocab().iter().map(|token| token.len()).sum::<usize>() as u64)
    );

    let tokenizer = Tokenizer::new(model, processors, special_tokens);
    tokenizer.save(output).unwrap();

    log::info!("Saved merged vocabulary to {:?}", output);
}

#[allow(clippy::too_many_arguments)]
fn mine_cmd(num_idioms: usize, output: &str, train: &[String], patterns: &[String]) {
    assert!(
        !train.is_empty(),
        "At least one train source must be provided."
    );
    assert!(
        !patterns.is_empty(),
        "At least one pattern must be provided."
    );

    log::info!(
        "Mining idioms output={:?} num_idioms={} patterns={:?}",
        output,
        num_idioms,
        patterns
    );

    let train = load_sources(train, &[], "train");
    let train_samples = shuffled_train_samples(&train);
    let patterns = load_patterns(patterns);
    let re = build_mine_regex(patterns);

    let idiom_miner = IdiomMiner::new(num_idioms, re);

    let idioms = idiom_miner.mine(&train_samples);

    log::info!("Found {} idioms.", idioms.len());

    for (idiom, count) in &idioms {
        log::debug!(
            "{:?}: {} (~{:.2} per sample)",
            idiom,
            count,
            (*count as f64) / (train_samples.len() as f64)
        );
    }

    let idioms = idioms
        .iter()
        .map(|(idiom, _)| idiom.clone())
        .collect::<Vec<String>>();

    std::fs::write(output, serde_json::to_string_pretty(&idioms).unwrap()).unwrap();
}

fn main() {
    env_logger::Builder::from_default_env().init();

    match flags::Tokengeex::from_env_or_exit().subcommand {
        flags::TokengeexCmd::Generate(flags) => {
            generate_cmd(
                // --- General Purpose ---
                &flags.output,
                flags.vocab_size,
                // --- Data ---
                &flags.train,
                // --- Processing ---
                &flags.processor,
                // --- Suggested, Added and Special Tokens ---
                &flags.special,
                &flags.suggested,
                &flags.added,
                // --- Options ---
                flags.split,
                flags.allow,
                flags.insert_probability.unwrap_or(0.1),
                flags.max_token_length.unwrap_or(24),
            );
        }
        flags::TokengeexCmd::Prune(flags) => {
            prune_cmd(
                // --- General Purpose ---
                &flags.input,
                &flags.output,
                flags.vocab_size,
                // --- Data ---
                &flags.train,
                // --- Options ---
                flags.dropout.unwrap_or(0.01),
                flags.shrink_factor.unwrap_or(0.8),
                flags.em_subiters.unwrap_or(1),
            )
        }
        flags::TokengeexCmd::Filter(flags) => {
            filter_cmd(
                // --- General Purpose ---
                &flags.input,
                &flags.output,
                flags.vocab_size.unwrap_or(0),
                // --- Options ---
                flags.min_score,
                flags.force.unwrap_or(false),
            )
        }
        flags::TokengeexCmd::Regex(flags) => {
            regex_cmd(
                // --- General Purpose ---
                &flags.output,
                // --- Options ---
                &flags.pattern,
            )
        }
        flags::TokengeexCmd::Merge(flags) => {
            merge_cmd(
                // --- General Purpose ---
                &flags.input,
                &flags.output,
                // --- Data ---
                &flags.train,
                // --- Options ---
                &flags.allow,
                flags.num_merges.unwrap_or(1000),
                flags.step.unwrap_or(50),
                flags.scale_factor.unwrap_or(0.9),
                flags.max_token_length.unwrap_or(24),
            )
        }
        flags::TokengeexCmd::Mine(flags) => {
            mine_cmd(
                // --- General Purpose ---
                flags.num_idioms,
                &flags.output,
                // --- Data ---
                &flags.train,
                // --- Options ---
                &flags.pattern,
            )
        }
        flags::TokengeexCmd::Encode(_) => {
            todo!();
        }
        flags::TokengeexCmd::Decode(_) => {
            todo!();
        }
    }
}
