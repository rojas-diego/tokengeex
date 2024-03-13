use crate::utils::capcode;
use crate::Processor;
use serde::{ser::SerializeStruct, Serialize};
use unicode_normalization::UnicodeNormalization;

/// Replaces occurences of \r\n by \n.
#[derive(Clone)]
pub struct CrlfProcessor;

impl Processor for CrlfProcessor {
    fn preprocess(&self, s: &str) -> String {
        s.replace("\r\n", "\n")
    }

    fn postprocess(&self, s: &str) -> String {
        s.into()
    }
}

impl Serialize for CrlfProcessor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut processor = serializer.serialize_struct("CrlfProcessor", 1)?;

        processor.serialize_field("type", "crlf")?;

        processor.end()
    }
}

impl<'de> serde::Deserialize<'de> for CrlfProcessor {
    fn deserialize<D>(deserializer: D) -> Result<CrlfProcessor, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct CrlfProcessorVisitor;

        impl<'de> serde::de::Visitor<'de> for CrlfProcessorVisitor {
            type Value = CrlfProcessor;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct CrlfProcessor")
            }

            fn visit_map<A>(self, mut map: A) -> Result<CrlfProcessor, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                while let Some(key) = map.next_key::<&str>()? {
                    match key {
                        "type" => {
                            let value = map.next_value::<String>()?;
                            if value != "crlf" {
                                return Err(serde::de::Error::unknown_variant(&value, &["crlf"]));
                            }
                        }
                        _ => {
                            let _: serde::de::IgnoredAny = map.next_value()?;
                        }
                    }
                }

                Ok(CrlfProcessor)
            }
        }

        deserializer.deserialize_struct("CrlfProcessor", &["type"], CrlfProcessorVisitor)
    }
}

/// Applies the capcode encoding to the input string.
#[derive(Clone)]
pub struct CapcodeProcessor;

impl Processor for CapcodeProcessor {
    fn preprocess(&self, s: &str) -> String {
        capcode::encode(s)
    }

    fn postprocess(&self, s: &str) -> String {
        capcode::decode(s)
    }
}

impl Serialize for CapcodeProcessor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut processor = serializer.serialize_struct("CapcodeProcessor", 1)?;

        processor.serialize_field("type", "capcode")?;

        processor.end()
    }
}

impl<'de> serde::Deserialize<'de> for CapcodeProcessor {
    fn deserialize<D>(deserializer: D) -> Result<CapcodeProcessor, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct CapcodeProcessorVisitor;

        impl<'de> serde::de::Visitor<'de> for CapcodeProcessorVisitor {
            type Value = CapcodeProcessor;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct CapcodeProcessor")
            }

            fn visit_map<A>(self, mut map: A) -> Result<CapcodeProcessor, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                while let Some(key) = map.next_key::<&str>()? {
                    match key {
                        "type" => {
                            let value = map.next_value::<String>()?;
                            if value != "capcode" {
                                return Err(serde::de::Error::unknown_variant(
                                    &value,
                                    &["capcode"],
                                ));
                            }
                        }
                        _ => {
                            let _: serde::de::IgnoredAny = map.next_value()?;
                        }
                    }
                }

                Ok(CapcodeProcessor)
            }
        }

        deserializer.deserialize_struct("CapcodeProcessor", &["type"], CapcodeProcessorVisitor)
    }
}

/// Unicode normalizer.
#[derive(Clone)]
pub enum UnicodeProcessor {
    Nfc,
    Nfd,
    Nfkc,
    Nfkd,
}

impl Processor for UnicodeProcessor {
    fn preprocess(&self, s: &str) -> String {
        match self {
            UnicodeProcessor::Nfc => s.nfc().collect::<String>(),
            UnicodeProcessor::Nfd => s.nfd().collect::<String>(),
            UnicodeProcessor::Nfkc => s.nfkc().collect::<String>(),
            UnicodeProcessor::Nfkd => s.nfkd().collect::<String>(),
        }
    }

    fn postprocess(&self, s: &str) -> String {
        s.into()
    }
}

impl Serialize for UnicodeProcessor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut processor = serializer.serialize_struct("UnicodeProcessor", 2)?;

        processor.serialize_field("type", "unicode")?;
        processor.serialize_field(
            "form",
            match self {
                UnicodeProcessor::Nfc => "nfc",
                UnicodeProcessor::Nfd => "nfd",
                UnicodeProcessor::Nfkc => "nfkc",
                UnicodeProcessor::Nfkd => "nfkd",
            },
        )?;

        processor.end()
    }
}

impl<'de> serde::Deserialize<'de> for UnicodeProcessor {
    fn deserialize<D>(deserializer: D) -> Result<UnicodeProcessor, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct UnicodeProcessorVisitor;

        impl<'de> serde::de::Visitor<'de> for UnicodeProcessorVisitor {
            type Value = UnicodeProcessor;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct UnicodeProcessor")
            }

            fn visit_map<A>(self, mut map: A) -> Result<UnicodeProcessor, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut form = None;

                while let Some(key) = map.next_key::<&str>()? {
                    match key {
                        "form" => {
                            form = Some(map.next_value::<String>()?);
                        }
                        _ => {
                            let _: serde::de::IgnoredAny = map.next_value()?;
                        }
                    }
                }

                let form = form.ok_or_else(|| serde::de::Error::missing_field("form"))?;

                Ok(match form.as_str() {
                    "nfc" => UnicodeProcessor::Nfc,
                    "nfd" => UnicodeProcessor::Nfd,
                    "nfkc" => UnicodeProcessor::Nfkc,
                    "nfkd" => UnicodeProcessor::Nfkd,
                    _ => {
                        return Err(serde::de::Error::unknown_variant(
                            &form,
                            &["nfc", "nfd", "nfkc", "nfkd"],
                        ))
                    }
                })
            }
        }

        deserializer.deserialize_struct(
            "UnicodeProcessor",
            &["type", "form"],
            UnicodeProcessorVisitor,
        )
    }
}
