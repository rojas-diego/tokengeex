use serde::{de::Visitor, ser::SerializeStruct, Deserialize, Deserializer, Serialize};

use crate::unigram::Unigram;

pub struct SpecialToken {
    pub value: String,
    pub id: u32,
}

pub trait Model {
    fn encode(&self, _: &str) -> Vec<u32>;

    fn decode(&self, _: &[u32]) -> String;

    fn token_to_id(&self, _: &str) -> Option<u32>;

    fn id_to_token(&self, _: u32) -> Option<String>;

    fn vocab_size(&self) -> usize;
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum ModelWrapper {
    Unigram(Unigram),
}

impl Model for ModelWrapper {
    fn encode(&self, input: &str) -> Vec<u32> {
        match self {
            ModelWrapper::Unigram(model) => model.encode(input),
        }
    }

    fn decode(&self, input: &[u32]) -> String {
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

#[derive(Clone)]
pub struct Tokenizer {
    model: ModelWrapper,
}

impl Tokenizer {
    pub fn encode(&self, input: &str) -> Vec<u32> {
        self.model.encode(input)
    }

    pub fn decode(&self, input: &[u32]) -> String {
        self.model.decode(input)
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.model.token_to_id(token)
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.model.id_to_token(id)
    }

    pub fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }

    pub fn save(&self, filepath: &str) -> Result<(), Box<dyn std::error::Error>> {
        let contents = serde_json::to_string(self)?;
        std::fs::write(filepath, contents)?;
        Ok(())
    }
}

impl From<Unigram> for Tokenizer {
    fn from(model: Unigram) -> Self {
        Tokenizer {
            model: ModelWrapper::Unigram(model),
        }
    }
}

/// Load a tokenizer from a file.
pub fn load(file: &str) -> Result<Tokenizer, Box<dyn std::error::Error>> {
    let contents = std::fs::read_to_string(file)?;
    let tokenizer: Tokenizer = serde_json::from_str(&contents)?;
    Ok(tokenizer)
}

static SERIALIZATION_VERSION: &str = "1.0";

impl Serialize for Tokenizer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut tokenizer = serializer.serialize_struct("Tokenizer", 2)?;

        tokenizer.serialize_field("version", SERIALIZATION_VERSION)?;
        tokenizer.serialize_field("model", &self.model)?;

        tokenizer.end()
    }
}

impl<'de> Deserialize<'de> for Tokenizer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct("Tokenizer", &["version", "model"], TokenizerVisitor)
    }
}

struct TokenizerVisitor;

impl<'de> Visitor<'de> for TokenizerVisitor {
    type Value = Tokenizer;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("struct Tokenizer")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let mut version: Option<String> = None;
        let mut model: Option<ModelWrapper> = None;

        while let Some(key) = map.next_key()? {
            match key {
                "version" => {
                    version = Some(map.next_value()?);
                }
                "model" => {
                    model = Some(map.next_value()?);
                }
                _ => {
                    let _: serde::de::IgnoredAny = map.next_value()?;
                }
            }
        }

        match version {
            Some(v) if v == SERIALIZATION_VERSION => Ok(Tokenizer {
                model: model.expect("model not found"),
            }),
            Some(v) => Err(serde::de::Error::custom(format!(
                "unsupported version: {}",
                v
            ))),
            None => Err(serde::de::Error::missing_field("version")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_deserialize() {
        let tokenizer_json = r#"{"version":"1.0","model":{"type":"unigram"}}"#;
        let tokenizer: Result<Tokenizer, _> = serde_json::from_str(tokenizer_json);
        assert!(tokenizer.is_ok());

        let tokenizer_json = r#"{"version":"2.0","model":{"type":"unigram"}}"#;
        let tokenizer: Result<Tokenizer, _> = serde_json::from_str(tokenizer_json);
        assert!(tokenizer.is_err());

        let tokenizer_json = r#"{"version":"1.0","model":{"type":"bigram"}}"#;
        let tokenizer: Result<Tokenizer, _> = serde_json::from_str(tokenizer_json);
        assert!(tokenizer.is_err());
    }
}
