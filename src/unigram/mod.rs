use serde::{
    de::{MapAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Serialize, Serializer,
};

use crate::core::Model;

mod lattice;
mod trie;

pub struct Unigram;

impl Model for Unigram {
    fn encode(&self, _: &str) -> Vec<u32> {
        unimplemented!()
    }

    fn decode(&self, _: &[u32]) -> String {
        unimplemented!()
    }

    fn token_to_id(&self, _: &str) -> Option<u32> {
        unimplemented!()
    }

    fn id_to_token(&self, _: u32) -> Option<String> {
        unimplemented!()
    }

    fn vocab_size(&self) -> usize {
        unimplemented!()
    }
}

impl Serialize for Unigram {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut model = serializer.serialize_struct("Unigram", 1)?;

        model.serialize_field("type", "unigram")?;

        model.end()
    }
}

impl<'de> Deserialize<'de> for Unigram {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_struct("Unigram", &[], UnigramVisitor)
    }
}

struct UnigramVisitor;

impl<'de> Visitor<'de> for UnigramVisitor {
    type Value = Unigram;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("struct Unigram")
    }

    fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
    where
        V: MapAccess<'de>,
    {
        let mut model_type: Option<String> = None;

        while let Some(key) = map.next_key()? {
            match key {
                "type" => {
                    model_type = Some(map.next_value()?);
                }
                _ => {
                    return Err(serde::de::Error::unknown_field(key, &["type"]));
                }
            }
        }

        match model_type.as_deref() {
            Some("unigram") => Ok(Unigram),
            _ => Err(serde::de::Error::custom("invalid model type")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn serialize_unigram() {
        let unigram = Unigram;
        let serialized = serde_json::to_string(&unigram).unwrap();
        assert_eq!(serialized, r#"{"type":"unigram"}"#);
    }

    #[test]
    fn deserialize_unigram() {
        let deserialized: Result<Unigram, serde_json::Error> =
            serde_json::from_str(r#"{"type":"unigram"}"#);
        assert!(deserialized.is_ok());

        let deserialized: Result<Unigram, serde_json::Error> =
            serde_json::from_str(r#"{"type":"bigram"}"#);
        assert!(deserialized.is_err());
    }
}
