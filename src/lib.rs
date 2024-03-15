pub mod unigram;

mod model;
mod processor;
mod tokenizer;
mod utils;
mod vocab;

pub use model::*;
pub use processor::*;
pub use tokenizer::*;
pub use utils::*;
pub use vocab::*;

/// The maximum length of a token in bytes.
pub const MAX_TOKEN_LENGTH: usize = 63;

/// An arbitrary sequence of bytes. Almost always valid UTF-8 but not
/// guaranteed.
/// Never longer than `MAX_TOKEN_LENGTH`.
pub type Token = Vec<u8>;

/// A token and its score.
pub type ScoredToken = (Token, f64);

pub type Error = Box<dyn std::error::Error + Send>;
pub type Result<T> = std::result::Result<T, Error>;

/// Load a tokenizer from a file.
pub fn load(file: &str) -> std::result::Result<Tokenizer, Box<dyn std::error::Error>> {
    let contents = std::fs::read_to_string(file)?;
    let tokenizer: Tokenizer = serde_json::from_str(&contents)?;
    Ok(tokenizer)
}
