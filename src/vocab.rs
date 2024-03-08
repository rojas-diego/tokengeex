use std::collections::{HashMap, HashSet};

use rand::Rng;
use regex::Regex;

pub const STRICT_RE: &str = r#"(?:^.$)|(?:^[[:punct:][:space:][DCU]]+$)|(?:^[\u3400-\u4DBF\u4E00-\u9FFF]+$)|(?:^(?:D?[UC]?)?(?: ?(?:(?:[a-z]+|[0-9]{1,4})(?:D?[UC]?))){0,4}$)|(?:^<D?[UC]? [a-z]+(?:>|/>| />)?$)"#;
pub const BASE_RE: &str = r#"(?:^.$)|(?:^[[:punct:][:space:][DCU]]+$)|(?:^[\u3400-\u4DBF\u4E00-\u9FFF]+$)|(?:^(?:D?[UC]?)?(?: ?(?:(?:[a-z\._]+|[0-9]{1,4})(?:D?[UC]?))){0,4}$)|(?:^<D?[UC]? [a-z]+(?:>|/>| />)?$)"#;
pub const ADVANCED_RE: &str = r#"(?:^.$)|(?:^[[:punct:][:space:][DCU]]+$)|(?:^[\u3400-\u4DBF\u4E00-\u9FFF]+$)|(?:^(?:D?[UC]?)?(?: ?(?:(?:[a-z\._:/\-\*]+|[0-9]{1,4})(?:D?[UC]?))){0,4}$)|(?:^<D?[UC]? [a-z]+(?:>|/>| />)?$)"#;

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
        let strict = 0b00000100;
        let base: u8 = 0b00000010;
        let advanced = 0b00000001;
        let none: u8 = 0;
        let all: u8 = strict | base | advanced;

        // An array of samples and whether it is a valid token for each
        // configuration.
        // ({token}, {bit_mask})
        let samples = [
            ("word", all),
            (" word", all),
            ("word ", none),
            ("two words", all),
            (" in order to", all),
            ("123", all),
            (" 456", all),
            ("789 ", none),
            ("好", all),
            ("你好", all),
            ("我叫罗杰斯", all),
            ("DC complexDU casingD 123", all),
            ("1.D 0", base | advanced),
            (" 2.D 0", base | advanced),
            (" 150.D 0", base | advanced),
            (" 4.D 95", base | advanced),
            (" users_D table", base | advanced),
            ("1.D 0", base | advanced),
            ("D https://D github.D com", advanced),
            ("<D a>", all),
            ("<DU a", all),
            ("<D a/>", all),
            ("<D a />", all),
        ];

        let regexes = [
            ("strict", Regex::new(STRICT_RE).unwrap(), strict),
            ("base", Regex::new(BASE_RE).unwrap(), base),
            ("advanced", Regex::new(ADVANCED_RE).unwrap(), advanced),
        ];

        for (token, token_mask) in samples.iter() {
            for (name, re, re_mask) in regexes.iter() {
                let should_match: bool = (token_mask & re_mask) != 0;
                assert!(
                    re.is_match(token) == should_match,
                    "Expected {:?} {}to match {} Regex ({:?}",
                    token,
                    match should_match {
                        false => "not ",
                        true => "",
                    },
                    name,
                    re
                );
            }
        }
    }
}
