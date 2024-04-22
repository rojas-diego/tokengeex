use ::regex::Regex;
use fancy_regex::Regex as FancyRegex;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tokengeex::{CrlfProcessor, Model, Processor, ProcessorWrapper, Tokenizer, UnicodeProcessor};

mod filter;
mod generate;
mod prune;
mod regex;

pub use filter::*;
pub use generate::*;
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
                repeated --special-token special_token: String

                // --- Options ---
                /// A Regex rule. If specified, only substrings that
                /// match this regex will be considered in the vocabulary.
                /// Does not support fancy regex syntax.
                optional --allow allow: String
                /// A Regex rule. If specified, when constructing the
                /// vocabulary, each sample will be according to this regex.
                /// Supports fancy regex syntax.
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
                /// IDs of the tokens to remove.
                repeated --id id: u32
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

                // --- Options ---
                /// The set of rules that define what tokens can be merged.
                repeated --allow allow: String
                /// The number of merges to perform.
                optional --num-merges num_merges: usize
                /// Step size for the BPE merge operations.
                optional --step step: usize
                /// Score scale factor.
                optional --score-scale-factor score_scale_factor: f64
                /// Maximum size of a token.
                optional --max-token-length max_token_length: usize
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

            /// Generate a Regex for downstream use with TokenGeeX.
            cmd regex {
                /// Output file to save the Regex.
                required -o, --output output: String
                /// Comma separated list of idioms to use.
                repeated -i, --idiom idiom: String
                /// List of Regex rules to use in addition to the idioms.
                repeated -r, --rule rule: String
            }
        }
    }
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
    let split_regex = split.map(|split| {
        FancyRegex::new(
            std::fs::read_to_string(split)
                .unwrap()
                .replace(['\n', '\r'], "")
                .trim(),
        )
        .unwrap()
    });
    let allow_regex = allow.map(|allow| {
        Regex::new(
            std::fs::read_to_string(allow)
                .unwrap()
                .replace(['\n', '\r'], "")
                .trim(),
        )
        .unwrap()
    });

    log::debug!("Allow regex: {:?}", allow_regex);
    log::debug!("Split regex: {:?}", split_regex);

    let mut vocab_generator = VocabularyGenerator::new(
        max_token_length,
        insert_probability,
        split_regex,
        allow_regex,
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
    shrink_factor: f64,
    em_subiters: usize,
) {
    log::info!(
        "Pruning vocabulary input={:?} output={:?} vocab_size={} shrink_factor={} em_subiters={}",
        input,
        output,
        vocab_size,
        shrink_factor,
        em_subiters
    );

    let (mut model, processors, special_tokens) = Tokenizer::from_file(input).unwrap().into_inner();
    let prev_vocab_size = model.vocab_size();
    let train = load_sources(train, &processors, "train");
    let train_samples = train
        .iter()
        .flat_map(|source| source.processed_samples.iter())
        .map(|s| s.as_str())
        .collect::<Vec<&str>>();

    let vocab_pruner = ModelVocabularyPruner::new(vocab_size, shrink_factor, em_subiters);

    vocab_pruner.prune(&mut model, &train_samples).unwrap();

    log::info!(
        "Pruned vocabulary from={} to={} mem={}",
        prev_vocab_size,
        vocab_size,
        format_bytes_as_mb(model.vocab().iter().map(|token| token.len()).sum::<usize>() as u64)
    );

    let tokenizer = Tokenizer::new(model, processors, special_tokens);
    tokenizer.save(output).unwrap();

    log::info!("Saved pruned vocabulary to {:?}", output);
}

#[allow(clippy::too_many_arguments)]
fn filter_cmd(
    input: &str,
    output: &str,
    vocab_size: usize,
    ids: &[u32],
    min_score: Option<f64>,
    force: bool,
) {
    log::info!(
        "Filtering vocabulary input={:?} output={:?} vocab_size={} ids={} min_score={:?} force={}",
        input,
        output,
        vocab_size,
        ids.len(),
        min_score,
        force
    );

    let (mut model, processors, special_tokens) = Tokenizer::from_file(input).unwrap().into_inner();

    let vocab_filter = VocabularyFilter::new(vocab_size, ids, min_score, force);

    vocab_filter.filter(&mut model);

    let tokenizer = Tokenizer::new(model, processors, special_tokens);
    tokenizer.save(output).unwrap();

    log::info!("Saved filtered vocabulary to {:?}", output);
}

#[allow(clippy::too_many_arguments)]
fn regex_cmd(output: &str, idioms: &[String], rules: &[String]) {
    log::info!(
        "Generating regex output={:?} idioms={:?} rules={:?}",
        output,
        idioms.len(),
        rules
    );

    let idioms = idioms
        .iter()
        .map(|name| {
            IDIOMS
                .iter()
                .find(|(n, _, _, _)| n == name)
                .unwrap_or_else(|| panic!("Idiom {:?} not found.", name))
                .1
                .to_string()
        })
        .collect::<Vec<String>>();

    let rules = idioms
        .iter()
        .chain(rules.iter())
        .map(|rule| {
            Regex::new(rule)
                .unwrap_or_else(|e| panic!("Failed to compile regex {:?}: {:?}", rule, e))
        })
        .collect::<Vec<Regex>>();

    let regexes = build_allow_regex(rules);

    log::debug!("Generated regex: {:?}", regexes);

    std::fs::write(output, regexes.as_str()).unwrap();

    log::info!("Saved regex to {:?}", output);
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
                &flags.special_token,
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
                &flags.id,
                flags.min_score,
                flags.force.unwrap_or(false),
            )
        }
        flags::TokengeexCmd::Regex(flags) => {
            regex_cmd(
                // --- General Purpose ---
                &flags.output,
                // --- Options ---
                &flags.idiom,
                &flags.rule,
            )
        }
        flags::TokengeexCmd::Merge(_) => {
            todo!();
        }
        flags::TokengeexCmd::Encode(_) => {
            todo!();
        }
        flags::TokengeexCmd::Decode(_) => {
            todo!();
        }
    }
}
