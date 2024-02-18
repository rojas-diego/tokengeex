use std::marker::PhantomData;

use serde::{
    de::{DeserializeOwned, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize,
};

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

pub struct Tokenizer<M> {
    model: M,
}

impl<M> Tokenizer<M>
where
    M: Model + DeserializeOwned + Serialize,
{
    pub fn new(model: M) -> Tokenizer<M> {
        Tokenizer { model }
    }

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

/// Load a tokenizer from a file.
pub fn load<M>(file: &str) -> Result<Tokenizer<M>, Box<dyn std::error::Error>>
where
    M: Model + DeserializeOwned + Serialize,
{
    let contents = std::fs::read_to_string(file)?;
    let tokenizer: Tokenizer<M> = serde_json::from_str(&contents)?;
    Ok(tokenizer)
}

static SERIALIZATION_VERSION: &str = "1.0";

impl<M> Serialize for Tokenizer<M>
where
    M: Model + DeserializeOwned + Serialize,
{
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

impl<'de, M> Deserialize<'de> for Tokenizer<M>
where
    M: Model + DeserializeOwned + Serialize,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "Tokenizer",
            &["version", "model"],
            TokenizerVisitor(PhantomData),
        )
    }
}

struct TokenizerVisitor<M>(PhantomData<M>);

impl<'de, M> Visitor<'de> for TokenizerVisitor<M>
where
    M: Model + DeserializeOwned + Serialize,
{
    type Value = Tokenizer<M>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("struct Tokenizer")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let mut version: Option<String> = None;
        let mut model: Option<M> = None;

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
    use crate::unigram::Unigram;

    use super::*;
    use serde_json;

    #[test]
    fn test_deserialize() {
        let tokenizer_json = r#"{"version":"1.0","model":{"type":"unigram"}}"#;
        let tokenizer: Result<Tokenizer<Unigram>, _> = serde_json::from_str(tokenizer_json);
        assert!(tokenizer.is_ok());

        let tokenizer_json = r#"{"version":"2.0","model":{"type":"unigram"}}"#;
        let tokenizer: Result<Tokenizer<Unigram>, _> = serde_json::from_str(tokenizer_json);
        assert!(tokenizer.is_err());

        let tokenizer_json = r#"{"version":"1.0","model":{"type":"bigram"}}"#;
        let tokenizer: Result<Tokenizer<Unigram>, _> = serde_json::from_str(tokenizer_json);
        assert!(tokenizer.is_err());
    }
}
