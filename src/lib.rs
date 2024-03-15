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

#[derive(Debug)]
pub struct LoadError {
    inner: Box<dyn std::error::Error>,
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "failed to load tokenizer")
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&*self.inner)
    }
}

impl From<LoadError> for Box<dyn std::error::Error + Send> {
    fn from(e: LoadError) -> Box<dyn std::error::Error + Send> {
        e.into()
    }
}

/// Load a tokenizer from a file.
pub fn load(file: &str) -> Result<Tokenizer> {
    let file = std::fs::File::open(file).map_err(|e| LoadError { inner: Box::new(e) })?;
    let tokenizer = serde_json::from_reader(file).map_err(|e| LoadError { inner: Box::new(e) })?;

    Ok(tokenizer)
}
