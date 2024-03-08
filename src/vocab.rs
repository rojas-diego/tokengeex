use std::collections::{HashMap, HashSet};

use rand::Rng;
use regex::Regex;

pub struct VocabularyGenerator {
    max_token_length: usize,
    insert_probability: f64,
    allow: Vec<Regex>,
    disallow: Vec<Regex>,
}

impl VocabularyGenerator {
    pub fn new(
        max_token_length: usize,
        insert_probability: f64,
        allow: &[String],
        disallow: &[String],
    ) -> VocabularyGenerator {
        let allow = allow.iter().map(|s| Regex::new(s).unwrap()).collect();
        let disallow = disallow.iter().map(|s| Regex::new(s).unwrap()).collect();

        VocabularyGenerator {
            max_token_length,
            insert_probability,
            allow,
            disallow,
        }
    }

    /// Collect frequent tokens from an array of samples.
    pub fn collect_frequent_tokens<'a>(
        &self,
        samples: impl Iterator<Item = &'a str>,
    ) -> HashMap<String, usize> {
        let mut frequencies = HashMap::new();
        let mut rng = rand::thread_rng();

        for sample in samples {
            let mut sample_tokens = HashSet::new();

            for (i, _) in sample.char_indices() {
                let suffix = &sample[i..];
                for (ii, c) in suffix.char_indices().take(self.max_token_length) {
                    let candidate = &suffix[..ii + c.len_utf8()];

                    if !self.disallow.iter().any(|re| re.is_match(candidate))
                        && (self.allow.iter().any(|re| re.is_match(candidate))
                            || self.allow.is_empty())
                        && rng.gen_range(0.0..1.0) < self.insert_probability
                    {
                        sample_tokens.insert(candidate);
                    }
                }
            }

            for token in sample_tokens {
                *frequencies.entry(token.to_string()).or_insert(0) += 1;
            }
        }

        frequencies
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regexes() {
        let re = Regex::new(r#"^(?:.|\s| ?(?:[DUC]+) ?| ?[a-z]+(?: [a-z]+){0,2}| ?[0-9]{0,3})$"#)
            .unwrap();
        let valid = vec![
            "hello",
            "hello world",
            "hello world again",
            " hello",
            " DU",
            " DC",
            " D ",
            "DU ",
            "DC ",
            "D ",
            " 1",
            " 12",
            "9",
            "########",
            "987",
        ];
        let invalid = vec![
            "hello world again and again",
            "Hello",
            "hello world!",
            "1234",
        ];

        for s in valid {
            assert!(re.is_match(s), "Regex {:?} failed to match: {}", re, s);
        }
        for s in invalid {
            assert!(!re.is_match(s), "Regex {:?} matched: {}", re, s);
        }
    }
}
