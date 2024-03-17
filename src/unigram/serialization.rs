use super::{ScoredToken, Unigram};
use base64::{engine::general_purpose::STANDARD_NO_PAD as BASE64_STANDARD, Engine};
use serde::{
    de::{MapAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Serialize, Serializer,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
/// A serialized token is a token and an optional flag to indicate whether the
/// token is base64 encoded. This is used to ensure that the JSON vocabulary
/// file is still human readable even in the presence of invalid Unicode tokens.
struct SerializedScoredToken {
    value: String,
    score: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoded: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// A vocabulary struct, used only for serialization and deserialization.
pub struct Vocab(Vec<SerializedScoredToken>);

impl From<Vec<ScoredToken>> for Vocab {
    fn from(vocab: Vec<ScoredToken>) -> Self {
        let mut serialized_vocab = Vec::with_capacity(vocab.len());

        for (token, score) in vocab {
            let mut encoded = None;
            let value = String::from_utf8(token.to_vec()).unwrap_or_else(|_| {
                encoded = Some(true);
                BASE64_STANDARD.encode(token.as_slice())
            });

            serialized_vocab.push(SerializedScoredToken {
                value,
                score,
                encoded,
            });
        }

        Vocab(serialized_vocab)
    }
}

impl From<Vocab> for Vec<ScoredToken> {
    fn from(vocab: Vocab) -> Self {
        vocab
            .0
            .into_iter()
            .map(
                |SerializedScoredToken {
                     value,
                     score,
                     encoded,
                 }| {
                    let token = if let Some(true) = encoded {
                        BASE64_STANDARD.decode(value.as_bytes()).unwrap()
                    } else {
                        value.as_bytes().to_vec()
                    };

                    (token[..].into(), score)
                },
            )
            .collect()
    }
}

type StdResult<T, E> = std::result::Result<T, E>;

impl Serialize for Unigram {
    fn serialize<S>(&self, serializer: S) -> StdResult<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut model = serializer.serialize_struct("Unigram", 1)?;
        let serialized_vocab = Vocab::from(self.vocab.clone());

        model.serialize_field("type", "unigram")?;
        model.serialize_field("vocab", &serialized_vocab)?;

        model.end()
    }
}

impl<'de> Deserialize<'de> for Unigram {
    fn deserialize<D>(deserializer: D) -> StdResult<Self, D::Error>
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

    fn visit_map<V>(self, mut map: V) -> StdResult<Self::Value, V::Error>
    where
        V: MapAccess<'de>,
    {
        let mut model_type: Option<String> = None;
        let mut serialized_vocab: Option<Vocab> = None;

        while let Some(key) = map.next_key()? {
            match key {
                "type" => {
                    model_type = Some(map.next_value()?);
                }
                "vocab" => {
                    serialized_vocab = Some(map.next_value()?);
                }
                _ => {
                    return Err(serde::de::Error::unknown_field(key, &["type"]));
                }
            }
        }

        if model_type.is_none() {
            return Err(serde::de::Error::missing_field("type"));
        }

        if model_type.as_deref() != Some("unigram") {
            return Err(serde::de::Error::custom("invalid model type"));
        }

        let vocab: Vec<ScoredToken> = serialized_vocab
            .ok_or_else(|| serde::de::Error::missing_field("vocab"))?
            .into();

        Ok(Unigram::from(vocab))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_serialize_unigram() {
        let vocab = [("a", 1.0), ("b", 2.0), ("c", 3.0)]
            .iter()
            .map(|(s, f)| (s.into(), *f))
            .collect();

        let model = Unigram::from(vocab);

        let serialized = serde_json::to_string(&model).unwrap();
        let deserialized: Unigram = serde_json::from_str(&serialized).unwrap();

        assert_eq!(model.vocab, deserialized.vocab);
    }

    #[test]
    fn test_serialize_unigram_base64() {
        let vocab = [("a", 1.0), ("b", 2.0), ("c", 3.0)]
            .iter()
            .map(|(s, f)| (s.into(), *f))
            .collect();

        let mut model = Unigram::from(vocab);
        model.vocab.push((0x80.into(), 4.0));

        let serialized = serde_json::to_string(&model).unwrap();
        let deserialized: Unigram = serde_json::from_str(&serialized).unwrap();

        assert_eq!(model.vocab, deserialized.vocab);
    }

    #[test]
    fn test_serialize_deszerialize_invariants() {
        let vocab = (0..255_u8).map(|b| (b.into(), 1.0)).collect::<Vec<_>>();
        let model = Unigram::from(vocab);

        let serialized = serde_json::to_string(&model).unwrap();
        let deserialized: Unigram = serde_json::from_str(&serialized).unwrap();

        assert_eq!(model.vocab, deserialized.vocab);
    }
}
