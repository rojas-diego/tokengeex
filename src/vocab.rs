use std::collections::{HashMap, HashSet};

use rand::Rng;
use regex::Regex;

pub const STRICT_RE: &str = r#"(?:^.$)|(?:^[[:punct:][:space:][DCU]]+$)|(?:^[\u3400-\u4DBF\u4E00-\u9FFF]+$)|(?:^(?:D?[UC]?)?(?: ?(?:(?:[a-z]+|[0-9]{1,4})(?:D?[UC]?))){0,4}$)|(?:^<D?[UC]? [a-z]+(?:>|/>| />)?$)"#;
pub const BASE_RE: &str = r#"(?:^.$)|(?:^[[:punct:][:space:][DCU]]+$)|(?:^[\u3400-\u4DBF\u4E00-\u9FFF]+$)|(?:^(?:D?[UC]?)?(?: ?(?:(?:[a-z\._]+|[0-9]{1,4})(?:D?[UC]?))){0,4}$)|(?:^<D?[UC]? [a-z]+(?:>|/>| />)?$)"#;
pub const ADVANCED_RE: &str = r#"(?:^.$)|(?:^[[:punct:][:space:][DCU]]+$)|(?:^[\u3400-\u4DBF\u4E00-\u9FFF]+$)|(?:^(?:D?[UC]?)?(?: ?(?:(?:[a-z\._:/\-\*]+|[0-9]{1,4})(?:D?[UC]?))){0,4}$)|(?:^<D?[UC]? [a-z]+(?:>|/>| />)?$)"#;

pub struct VocabularyGenerator {
    max_token_length: usize,
    insert_probability: f64,
    allow: Regex,
}

impl VocabularyGenerator {
    pub fn new(
        max_token_length: usize,
        insert_probability: f64,
        allow: &str,
    ) -> VocabularyGenerator {
        let allow = Regex::new(allow).unwrap();

        VocabularyGenerator {
            max_token_length,
            insert_probability,
            allow,
        }
    }

    /// Collect frequent tokens from a sample.
    pub fn collect_frequent_tokens<'a>(
        &self,
        samples: impl IntoIterator<Item = &'a str>,
    ) -> HashMap<&'a str, usize> {
        let thread_local_allow = self.allow.clone();
        let mut frequent_tokens = HashMap::new();
        let mut rng = rand::thread_rng();

        for sample in samples {
            let mut sample_frequent_tokens = HashSet::new();

            for (i, _) in sample.char_indices() {
                let suffix = &sample[i..];
                for (ii, c) in suffix.char_indices().take(self.max_token_length) {
                    let candidate = &suffix[..ii + c.len_utf8()];

                    if thread_local_allow.is_match(candidate)
                        && rng.gen_range(0.0..1.0) < self.insert_probability
                    {
                        sample_frequent_tokens.insert(candidate);
                    }
                }
            }

            for token in sample_frequent_tokens {
                *frequent_tokens.entry(token).or_insert(0) += 1;
            }
        }

        frequent_tokens
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
