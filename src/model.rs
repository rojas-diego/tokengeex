use crate::{unigram, Result, ScoredToken, Token, TokenID};
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
    fn encode(&self, input: &str) -> Result<Vec<TokenID>>;

    /// Decode a sequence of token IDs into a string.
    ///
    /// # Errors
    ///
    /// This method can fail if the sequence of token IDs is invalid.
    fn decode(&self, ids: &[TokenID]) -> Result<String>;

    /// Convert a token to its corresponding ID. Returns None if the token is
    /// not in the vocabulary.
    fn token_to_id(&self, token: Token) -> Option<TokenID>;

    /// Convert an ID to its corresponding token. Returns None if the ID is not
    /// in the vocabulary.
    fn id_to_token(&self, id: TokenID) -> Option<ScoredToken>;

    /// Get the size of the vocabulary.
    fn vocab_size(&self) -> usize;

    /// Get the vocabulary.
    fn vocab(&self) -> &[ScoredToken];

    /// Add a token to the vocabulary.
    fn add_tokens<I>(&mut self, tokens: I)
    where
        I: IntoIterator<Item = ScoredToken>;
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModelWrapper {
    Unigram(unigram::Unigram),
}

impl Model for ModelWrapper {
    fn encode(&self, input: &str) -> Result<Vec<TokenID>> {
        match self {
            ModelWrapper::Unigram(model) => model.encode(input),
        }
    }

    fn decode(&self, input: &[TokenID]) -> Result<String> {
        match self {
            ModelWrapper::Unigram(model) => model.decode(input),
        }
    }

    fn token_to_id(&self, token: Token) -> Option<TokenID> {
        match self {
            ModelWrapper::Unigram(model) => model.token_to_id(token),
        }
    }

    fn id_to_token(&self, id: TokenID) -> Option<ScoredToken> {
        match self {
            ModelWrapper::Unigram(model) => model.id_to_token(id),
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            ModelWrapper::Unigram(model) => model.vocab_size(),
        }
    }

    fn vocab(&self) -> &[ScoredToken] {
        match self {
            ModelWrapper::Unigram(model) => model.vocab(),
        }
    }

    fn add_tokens<I>(&mut self, tokens: I)
    where
        I: IntoIterator<Item = ScoredToken>,
    {
        match self {
            ModelWrapper::Unigram(model) => model.add_tokens(tokens),
        }
    }
}
