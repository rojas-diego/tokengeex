use std::collections::HashMap;

use serde::{de::Visitor, ser::SerializeStruct, Deserialize, Deserializer, Serialize};

use crate::{Model, ModelWrapper, Processor, ProcessorWrapper, Result};

#[derive(Clone)]
pub struct Tokenizer {
    model: ModelWrapper,
    processors: Vec<ProcessorWrapper>,
    special_tokens: Vec<String>,
    special_tokens_map: HashMap<String, u32>,
}

#[derive(Clone, Copy, Debug)]
pub enum TokenizerError {
    TokenIdOutOfBounds(u32),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            TokenizerError::TokenIdOutOfBounds(id) => {
                write!(f, "token id {} is out of bounds", id)
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

impl Tokenizer {
    // TODO: Find a workaround for this.
    #[allow(private_interfaces)]
    pub fn new(model: ModelWrapper, processors: Vec<ProcessorWrapper>) -> Self {
        Tokenizer {
            model,
            processors,
            special_tokens: Vec::new(),
            special_tokens_map: HashMap::new(),
        }
    }

    /// Add special tokens to this tokenizer. Special tokens are encoded before
    /// the rest of the tokenization pipeline. They are assigned IDs starting
    /// from the end of the vocabulary.
    pub fn add_special_tokens(&mut self, tokens: &[&str]) {
        for token in tokens {
            self.special_tokens_map
                .insert(token.to_string(), self.special_tokens.len() as u32);
            self.special_tokens.push(token.to_string());
        }
    }

    /// Encode the input sequence into an array of token IDs. Special tokens
    /// are encoded first and then the model takes care of the rest.
    pub fn encode(&self, input: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();

        for (substr, is_special) in SpecialTokenSplitter::new(input, self.special_tokens.as_slice())
        {
            if is_special {
                ids.push(
                    self.model.vocab_size() as u32
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

    /// Decode the input sequence from an array of token IDs.
    pub fn decode(&self, input: &[u32]) -> Result<String> {
        let mut input = input;
        let mut output = String::new();

        // Continuously find the first special token in the input and decode it
        loop {
            let next_special_token_idx = input
                .iter()
                .position(|&id| id >= self.model.vocab_size() as u32);

            match next_special_token_idx {
                Some(idx) => {
                    output.push_str(&self.model.decode(&input[..idx])?);
                    output.push_str(
                        self.special_tokens
                            .get((input[idx] - self.model.vocab_size() as u32) as usize)
                            .ok_or(TokenizerError::TokenIdOutOfBounds(input[idx]))?,
                    );

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

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        if let Some(id) = self.special_tokens_map.get(token) {
            return Some(self.model.vocab_size() as u32 + id);
        }
        self.model.token_to_id(token)
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        if id < self.model.vocab_size() as u32 {
            self.model.id_to_token(id)
        } else if (id - self.model.vocab_size() as u32) < self.special_tokens.len() as u32 {
            Some(self.special_tokens[(id - self.model.vocab_size() as u32) as usize].clone())
        } else {
            None
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.model.vocab_size() + self.special_tokens.len()
    }

    pub fn save(&self, filepath: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let contents = serde_json::to_string(self)?;
        std::fs::write(filepath, contents)?;
        Ok(())
    }

    pub fn model(&self) -> ModelWrapper {
        self.model.clone()
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

        tokenizer.add_special_tokens(&special_tokens);

        Ok(tokenizer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize() {
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
