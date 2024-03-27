use std::collections::HashMap;

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{de::Visitor, ser::SerializeStruct, Deserialize, Deserializer, Serialize};

use crate::{
    Model, ModelWrapper, Processor, ProcessorWrapper, Result, ScoredToken, Token, TokenID,
};

#[derive(Clone)]
pub struct Tokenizer {
    model: ModelWrapper,
    processors: Vec<ProcessorWrapper>,
    special_tokens: Vec<String>,
    special_tokens_map: HashMap<String, TokenID>,
}

#[derive(Clone, Copy, Debug)]
pub enum TokenizerError {
    TokenIdOutOfBounds(TokenID),
    InvalidJSON,
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            TokenizerError::TokenIdOutOfBounds(id) => {
                write!(f, "token id {} is out of bounds", id)
            }
            TokenizerError::InvalidJSON => {
                write!(f, "invalid JSON")
            }
        }
    }
}

impl std::error::Error for TokenizerError {}

impl From<TokenizerError> for Box<dyn std::error::Error + Send> {
    fn from(err: TokenizerError) -> Self {
        Box::new(err)
    }
}

impl From<serde_json::Error> for TokenizerError {
    fn from(_: serde_json::Error) -> Self {
        TokenizerError::InvalidJSON
    }
}

impl Tokenizer {
    // Create a new tokenizer with a model and a list of processors.
    pub fn new<M, I>(model: M, processors: I) -> Self
    where
        M: Into<ModelWrapper>,
        I: IntoIterator,
        I::Item: Into<ProcessorWrapper>,
    {
        Tokenizer {
            model: model.into(),
            processors: processors.into_iter().map(|p| p.into()).collect(),
            special_tokens: Vec::new(),
            special_tokens_map: HashMap::new(),
        }
    }

    /// Add special tokens to this tokenizer. Special tokens are encoded before
    /// the rest of the tokenization pipeline. They are assigned IDs starting
    /// from the end of the vocabulary. If the special token is already present,
    /// it is ignored.
    pub fn add_special_tokens<I>(&mut self, tokens: I)
    where
        I: IntoIterator,
        I::Item: AsRef<str> + ToString,
    {
        for token in tokens {
            if self.special_tokens_map.contains_key(token.as_ref()) {
                continue;
            }

            self.special_tokens_map
                .insert(token.to_string(), self.special_tokens.len() as TokenID);
            self.special_tokens.push(token.to_string());
        }
    }

    /// Add tokens to the underlying model.
    pub fn add_tokens<I>(&mut self, tokens: I)
    where
        I: IntoIterator<Item = ScoredToken>,
    {
        self.model.add_tokens(tokens);
    }

    /// Encode the input sequence into an array of token IDs. Special tokens
    /// are encoded first and then the model takes care of the rest.
    pub fn encode(&self, input: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();

        for (substr, is_special) in SpecialTokenSplitter::new(input, self.special_tokens.as_slice())
        {
            if is_special {
                ids.push(
                    self.model.vocab_size() as TokenID
                        + self
                            .special_tokens_map
                            .get(substr)
                            .expect("captured special token to be in the special tokens map"),
                );
            } else {
                let processed = self
                    .processors
                    .iter()
                    .fold(substr.to_string(), |s, p| p.preprocess(&s));

                ids.extend(self.model.encode(&processed)?);
            }
        }

        Ok(ids)
    }

    /// Encode the input sequence without special tokens.
    pub fn encode_ordinary(&self, input: &str) -> Result<Vec<u32>> {
        let processed = self
            .processors
            .iter()
            .fold(input.to_string(), |s, p| p.preprocess(&s));

        self.model.encode(&processed)
    }

    /// Encode multiple samples at once.
    pub fn encode_batch<I>(&self, inputs: I) -> Result<Vec<Vec<u32>>>
    where
        I: IntoParallelIterator,
        I::Item: AsRef<str>,
    {
        inputs
            .into_par_iter()
            .map(|s| self.encode(s.as_ref()))
            .collect()
    }

    /// Encode multiple samples at once without special tokens.
    pub fn encode_ordinary_batch<I>(&self, inputs: I) -> Result<Vec<Vec<u32>>>
    where
        I: IntoParallelIterator,
        I::Item: AsRef<str>,
    {
        inputs
            .into_par_iter()
            .map(|s| self.encode_ordinary(s.as_ref()))
            .collect()
    }

    /// Decode the input sequence from an array of token IDs.
    pub fn decode(&self, input: &[TokenID], include_special_tokens: bool) -> Result<String> {
        let mut input = input;
        let mut output = String::new();

        // Continuously find the first special token in the input and decode it
        loop {
            let next_special_token_idx = input
                .iter()
                .position(|&id| id >= self.model.vocab_size() as TokenID);

            match next_special_token_idx {
                Some(idx) => {
                    output.push_str(&self.model.decode(&input[..idx])?);

                    let special = self
                        .special_tokens
                        .get((input[idx] - self.model.vocab_size() as TokenID) as usize)
                        .ok_or(TokenizerError::TokenIdOutOfBounds(input[idx]))?;

                    if include_special_tokens {
                        output.push_str(special);
                    }

                    input = &input[idx + 1..];
                }
                None => {
                    let decoded = self.model.decode(input)?;

                    output.push_str(
                        &self
                            .processors
                            .iter()
                            .rev()
                            .fold(decoded, |s, p| p.postprocess(&s)),
                    );

                    break;
                }
            }
        }

        Ok(output)
    }

    pub fn decode_batch<I>(&self, inputs: I, include_special_tokens: bool) -> Result<Vec<String>>
    where
        I: IntoParallelIterator,
        I::Item: AsRef<[TokenID]>,
    {
        inputs
            .into_par_iter()
            .map(|s| self.decode(s.as_ref(), include_special_tokens))
            .collect()
    }

    pub fn special_token_to_id(&self, token: &str) -> Option<TokenID> {
        self.special_tokens_map
            .get(token)
            .map(|id| *id + self.model.vocab_size() as TokenID)
    }

    pub fn token_to_id(&self, token: Token) -> Option<TokenID> {
        self.model.token_to_id(token)
    }

    pub fn id_to_special_token(&self, id: TokenID) -> Option<String> {
        if id < self.model.vocab_size() as TokenID {
            return None;
        }

        let id = id - self.model.vocab_size() as TokenID;
        self.special_tokens.get(id as usize).cloned()
    }

    pub fn id_to_token(&self, id: TokenID) -> Option<ScoredToken> {
        self.model.id_to_token(id)
    }

    pub fn is_special(&self, id: TokenID) -> Option<bool> {
        if id < self.model.vocab_size() as TokenID {
            return Some(false);
        }

        let id = id - self.model.vocab_size() as TokenID;
        Some(self.special_tokens.get(id as usize).is_some())
    }

    pub fn special_tokens(&self) -> Vec<String> {
        self.special_tokens.clone()
    }

    pub fn vocab_size(&self) -> usize {
        self.model.vocab_size() + self.special_tokens.len()
    }

    pub fn save(&self, filepath: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let contents = serde_json::to_string(self)?;
        std::fs::write(filepath, contents)?;
        Ok(())
    }

    pub fn model(&self) -> &ModelWrapper {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut ModelWrapper {
        &mut self.model
    }

    pub fn processors(&self) -> &Vec<ProcessorWrapper> {
        &self.processors
    }

    pub fn processors_mut(&mut self) -> &mut Vec<ProcessorWrapper> {
        &mut self.processors
    }
}

impl std::str::FromStr for Tokenizer {
    type Err = crate::Error;

    fn from_str(s: &str) -> Result<Self> {
        serde_json::from_str(s).map_err(|_| TokenizerError::InvalidJSON.into())
    }
}

impl ToString for Tokenizer {
    fn to_string(&self) -> String {
        serde_json::to_string(self).expect("failed to serialize Tokenizer")
    }
}

struct SpecialTokenSplitter<'a> {
    input: &'a str,
    special_tokens: &'a [String],
    cursor: usize,
}

impl<'a> SpecialTokenSplitter<'a> {
    fn new(input: &'a str, special_tokens: &'a [String]) -> Self {
        SpecialTokenSplitter {
            input,
            special_tokens,
            cursor: 0,
        }
    }
}

impl<'a> Iterator for SpecialTokenSplitter<'a> {
    type Item = (&'a str, bool);

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.input.len() {
            return None;
        }

        let input = &self.input[self.cursor..];

        for (i, _) in input.char_indices() {
            let suffix = &input[i..];

            for special_token in self.special_tokens {
                if suffix.starts_with(special_token) {
                    if i > 0 {
                        self.cursor += i;
                        return Some((&input[..i], false));
                    }

                    self.cursor += special_token.len();
                    return Some((&input[..special_token.len()], true));
                }
            }
        }

        if self.cursor < self.input.len() {
            self.cursor = self.input.len();
        }

        Some((&input, false))
    }
}

static SERIALIZATION_VERSION: &str = "1.0";

type StdResult<T, E> = std::result::Result<T, E>;

impl Serialize for Tokenizer {
    fn serialize<S>(&self, serializer: S) -> StdResult<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut tokenizer = serializer.serialize_struct("Tokenizer", 4)?;

        tokenizer.serialize_field("version", SERIALIZATION_VERSION)?;
        tokenizer.serialize_field("special_tokens", &self.special_tokens)?;
        tokenizer.serialize_field("processors", &self.processors)?;
        tokenizer.serialize_field("model", &self.model)?;

        tokenizer.end()
    }
}

impl<'de> Deserialize<'de> for Tokenizer {
    fn deserialize<D>(deserializer: D) -> StdResult<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "Tokenizer",
            &["version", "special_tokens", "processors", "model"],
            TokenizerVisitor,
        )
    }
}

struct TokenizerVisitor;

impl<'de> Visitor<'de> for TokenizerVisitor {
    type Value = Tokenizer;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("struct Tokenizer")
    }

    fn visit_map<A>(self, mut map: A) -> StdResult<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let mut version: Option<String> = None;
        let mut model: Option<ModelWrapper> = None;
        let mut special_tokens: Vec<&str> = Vec::new();
        let mut processors: Vec<ProcessorWrapper> = Vec::new();

        while let Some(key) = map.next_key()? {
            match key {
                "version" => {
                    version = Some(map.next_value()?);
                }
                "special_tokens" => {
                    special_tokens = map.next_value()?;
                }
                "processors" => {
                    processors = map.next_value()?;
                }
                "model" => {
                    model = Some(map.next_value()?);
                }
                _ => {
                    let _: serde::de::IgnoredAny = map.next_value()?;
                }
            }
        }

        let version = version.ok_or_else(|| serde::de::Error::missing_field("version"))?;
        if version != SERIALIZATION_VERSION {
            return Err(serde::de::Error::custom(format!(
                "unsupported version: {}",
                version
            )));
        }

        let model = model.ok_or_else(|| serde::de::Error::missing_field("model"))?;
        let mut tokenizer = Tokenizer::new(model, processors);

        tokenizer.add_special_tokens(special_tokens);

        Ok(tokenizer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_deserialize() {
        let tokenizer_json = r#"{"version":"1.0","model":{"type":"unigram","vocab":[]}}"#;
        let tokenizer: StdResult<Tokenizer, _> = serde_json::from_str(tokenizer_json);
        assert!(tokenizer.is_ok());

        let tokenizer_json = r#"{"version":"2.0","model":{"type":"unigram","vocab":[]}}"#;
        let tokenizer: StdResult<Tokenizer, _> = serde_json::from_str(tokenizer_json);
        assert!(tokenizer.is_err());

        let tokenizer_json = r#"{"version":"1.0","model":{"type":"bigram","vocab":[]}}"#;
        let tokenizer: StdResult<Tokenizer, _> = serde_json::from_str(tokenizer_json);
        assert!(tokenizer.is_err());
    }

    #[test]
    fn test_special_tokens_splitter() {
        let cases = [
            (
                "<EOS>Hello<EOS>",
                vec![("<EOS>", true), ("Hello", false), ("<EOS>", true)],
                vec!["<EOS>", "random", "<EOS_2>"],
            ),
            (
                "randomstring",
                vec![("random", true), ("string", false)],
                vec!["<EOS>", "random", "<EOS_2>"],
            ),
            (
                "random<EOS_2>string",
                vec![("random", true), ("<EOS_2>", true), ("string", false)],
                vec!["<EOS>", "random", "<EOS_2>"],
            ),
            (
                "nospecialtokens",
                vec![("nospecialtokens", false)],
                vec!["<EOS>", "random", "<EOS_2>"],
            ),
            (
                "No special tokens",
                vec![("No special tokens", false)],
                vec![],
            ),
        ];

        for (input, expected, special_tokens) in cases.iter() {
            let special_tokens = special_tokens
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>();
            let actual = SpecialTokenSplitter::new(input, special_tokens.as_slice())
                .map(|(s, b)| (s.to_string(), b))
                .collect::<Vec<_>>();
            let expected = expected
                .iter()
                .map(|(s, b)| (s.to_string(), *b))
                .collect::<Vec<_>>();

            assert_eq!(actual, expected);
        }
    }
}
