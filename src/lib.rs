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

/// A numerical ID for a token. Cannot be larger than `u32::MAX`.
pub type TokenID = u32;

/// An arbitrary sequence of bytes. Almost always valid UTF-8 but not
/// guaranteed.
/// Never longer than `MAX_TOKEN_LENGTH`.
pub type Token = Vec<u8>;

/// A token and its score.
pub type ScoredToken = (Token, f64);

pub type Error = Box<dyn std::error::Error + Send>;
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub struct LoadError {
    reason: String,
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "failed to load tokenizer: {}", self.reason)
    }
}

impl std::error::Error for LoadError {}

impl From<LoadError> for Box<dyn std::error::Error + Send> {
    fn from(e: LoadError) -> Self {
        Box::new(e)
    }
}

/// Load a tokenizer from a file.
pub fn load(file: &str) -> Result<Tokenizer> {
    let contents = std::fs::read_to_string(file).map_err(|e| LoadError {
        reason: e.to_string(),
    })?;
    let tokenizer = serde_json::from_str(&contents).map_err(|e| LoadError {
        reason: e.to_string(),
    })?;

    Ok(tokenizer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load() {
        let tokenizer = load("data/unigram-65k.json").unwrap();
        assert_eq!(tokenizer.vocab_size(), 65536);

        assert!(load("doesnotexist").is_err());
    }
}
