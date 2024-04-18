mod lattice;
mod model;
mod processor;
mod regex;
mod task;
mod tokenizer;
mod trie;

use base64::{engine::general_purpose::STANDARD_NO_PAD as BASE64_STANDARD, Engine};
use serde::{ser::SerializeStruct, Deserialize, Deserializer, Serialize, Serializer};

pub use lattice::*;
pub use model::*;
pub use processor::*;
pub use regex::*;
pub use task::*;
pub use tokenizer::*;
pub use trie::*;

/// A numerical ID for a token. Cannot be larger than `u32::MAX`.
pub type TokenID = u32;

/// An arbitrary sequence of bytes. Almost always valid UTF-8 but not
/// guaranteed.
pub type Token = Vec<u8>;

/// A token and its score.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoredToken {
    pub value: Token,
    pub score: f64,
    pub keep: bool,
}

impl ScoredToken {
    pub fn new(value: Token, score: f64, keep: bool) -> Self {
        Self { value, score, keep }
    }

    pub fn from_str(value: &str, score: f64, keep: bool) -> Self {
        Self {
            value: value.as_bytes().to_vec(),
            score,
            keep,
        }
    }

    pub fn from_u8(value: u8, score: f64, keep: bool) -> Self {
        Self {
            value: vec![value],
            score,
            keep,
        }
    }

    pub fn clone_with_score(&self, score: f64) -> Self {
        Self {
            value: self.value.clone(),
            score,
            keep: self.keep,
        }
    }

    pub fn clone_with_keep(&self, keep: bool) -> Self {
        Self {
            value: self.value.clone(),
            score: self.score,
            keep,
        }
    }

    pub fn len(&self) -> usize {
        self.value.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl PartialOrd for ScoredToken {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl Serialize for ScoredToken {
    // If the `token` field is valid UTF-8, it will be serialized as a string.
    // Otherwise, it is base64 encoded and "encoded" is set to true.
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ScoredToken", 2)?;
        let mut encoded = false;
        let value = String::from_utf8(self.value.clone()).unwrap_or_else(|_| {
            encoded = true;
            BASE64_STANDARD.encode(&self.value)
        });

        state.serialize_field("value", &value)?;
        state.serialize_field("score", &self.score)?;
        if encoded {
            state.serialize_field("encoded", &true)?;
        }
        if self.keep {
            state.serialize_field("keep", &true)?;
        }

        state.end()
    }
}

impl<'de> Deserialize<'de> for ScoredToken {
    fn deserialize<D>(deserializer: D) -> std::result::Result<ScoredToken, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ScoredTokenVisitor;

        impl<'de> serde::de::Visitor<'de> for ScoredTokenVisitor {
            type Value = ScoredToken;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("ScoredToken")
            }

            fn visit_map<V>(self, mut map: V) -> std::result::Result<ScoredToken, V::Error>
            where
                V: serde::de::MapAccess<'de>,
            {
                let mut token: Option<String> = None;
                let mut score = None;
                let mut encoded = false;
                let mut keep = false;

                while let Some(key) = map.next_key()? {
                    match key {
                        "token" => {
                            token = map.next_value()?;
                        }
                        "score" => {
                            score = map.next_value()?;
                        }
                        "encoded" => {
                            encoded = map.next_value()?;
                        }
                        "keep" => {
                            keep = map.next_value()?;
                        }
                        _ => {
                            return Err(serde::de::Error::unknown_field(key, FIELDS));
                        }
                    }
                }

                let token = match token {
                    Some(token) => {
                        if encoded {
                            BASE64_STANDARD
                                .decode(token.as_bytes())
                                .map_err(serde::de::Error::custom)?
                        } else {
                            token.into_bytes()
                        }
                    }
                    None => return Err(serde::de::Error::missing_field("token")),
                };

                let score = match score {
                    Some(score) => score,
                    None => return Err(serde::de::Error::missing_field("score")),
                };

                Ok(ScoredToken {
                    value: token,
                    score,
                    keep,
                })
            }
        }

        const FIELDS: &[&str] = &["token", "score", "encoded", "keep"];
        deserializer.deserialize_struct("ScoredToken", FIELDS, ScoredTokenVisitor)
    }
}

pub fn new_default_vocab() -> Vec<ScoredToken> {
    (0..=255)
        .map(|id| ScoredToken::new(vec![id as u8], 1.0 / 256.0, false))
        .collect()
}

pub fn make_vocab(tokens: &[(&[u8], f64)]) -> Vec<ScoredToken> {
    tokens
        .iter()
        .map(|(token, score)| ScoredToken::new(token.to_vec(), *score, false))
        .collect()
}

pub enum Error {
    IO(std::io::Error),
    SerdeJSON(serde_json::Error),
    TokenIdOutOfBounds(TokenID),
    NoPath(usize, usize),
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::IO(err)
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::SerdeJSON(err)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::IO(err) => write!(f, "{}", err),
            Error::SerdeJSON(err) => write!(f, "{}", err),
            Error::NoPath(pos, len) => write!(f, "no path to position {}/{}", pos, len),
            Error::TokenIdOutOfBounds(id) => write!(f, "token id {} is out of bounds", id),
        }
    }
}

impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::IO(err) => write!(f, "IO({:?})", err),
            Error::SerdeJSON(err) => write!(f, "SerdeJSON({:?})", err),
            Error::NoPath(pos, len) => write!(f, "NoPath({}, {})", pos, len),
            Error::TokenIdOutOfBounds(id) => write!(f, "TokenIdOutOfBounds({})", id),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_scored_token() {
        let scored_token = ScoredToken::new(b"hello".to_vec(), 0.5, false);
        let serialized = serde_json::to_string(&scored_token).unwrap();
        let deserialized: ScoredToken = serde_json::from_str(&serialized).unwrap();
        assert_eq!(scored_token.value, deserialized.value);
        assert_eq!(scored_token.score, deserialized.score);
    }
}
