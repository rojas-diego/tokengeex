use crate::{unigram, Result};
use serde::{Deserialize, Serialize};

/// A model is a tokenization algorithm that can encode and decode sequences of
/// characters into sequences of token IDs and vice versa.
pub trait Model {
    /// Encode a string into a sequence of token IDs.
    ///
    /// # Errors
    ///
    /// This method can fail if the model is unable to encode the input string
    /// such as when the input string contains characters that are not in the
    /// model's vocabulary.
    fn encode(&self, s: &str) -> Result<Vec<u32>>;

    /// Decode a sequence of token IDs into a string.
    ///
    /// # Errors
    ///
    /// This method can fail if the sequence of token IDs is invalid.
    fn decode(&self, ids: &[u32]) -> Result<String>;

    /// Convert a token to its corresponding ID. Returns None if the token is
    /// not in the vocabulary.
    fn token_to_id(&self, token: &str) -> Option<u32>;

    /// Convert an ID to its corresponding token. Returns None if the ID is not
    /// in the vocabulary.
    fn id_to_token(&self, id: u32) -> Option<String>;

    /// Get the size of the vocabulary.
    fn vocab_size(&self) -> usize;
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModelWrapper {
    Unigram(unigram::Unigram),
}

impl Model for ModelWrapper {
    fn encode(&self, input: &str) -> Result<Vec<u32>> {
        match self {
            ModelWrapper::Unigram(model) => model.encode(input),
        }
    }

    fn decode(&self, input: &[u32]) -> Result<String> {
        match self {
            ModelWrapper::Unigram(model) => model.decode(input),
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            ModelWrapper::Unigram(model) => model.token_to_id(token),
        }
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        match self {
            ModelWrapper::Unigram(model) => model.id_to_token(id),
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            ModelWrapper::Unigram(model) => model.vocab_size(),
        }
    }
}
