use crate::{logprobs, parallelism::*, ScoredToken};
use dashmap::DashMap;
use rand::Rng;
use regex::Regex;
use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet},
};

pub const STRICT_RE: &str = r#"(?:^.$)|(?:^[[:punct:][:space:][DCU]]+$)|(?:^[\u3400-\u4DBF\u4E00-\u9FFF]+$)|(?:^(?:D?[UC]?)?(?: ?(?:(?:[a-z]+|[0-9]{1,4})(?:D?[UC]?))){0,4}$)|(?:^<D?[UC]? [a-z]+(?:>|/>| />)?$)"#;
pub const BASE_RE: &str = r#"(?:^.$)|(?:^[[:punct:][:space:][DCU]]+$)|(?:^[\u3400-\u4DBF\u4E00-\u9FFF]+$)|(?:^(?:D?[UC]?)?(?: ?(?:(?:[a-z\._]+|[0-9]{1,4})(?:D?[UC]?))){0,4}$)|(?:^<D?[UC]? [a-z]+(?:>|/>| />)?$)"#;
pub const ADVANCED_RE: &str = r#"(?:^.$)|(?:^[[:punct:][:space:][DCU]]+$)|(?:^[\u3400-\u4DBF\u4E00-\u9FFF]+$)|(?:^(?:D?[UC]?)?(?: ?(?:(?:[a-z\._:/\-\*]+|[0-9]{1,4})(?:D?[UC]?))){0,4}$)|(?:^<D?[UC]? [a-z]+(?:>|/>| />)?$)"#;

pub struct VocabularyGenerator {
    max_token_length: usize,
    insert_probability: f64,
    allow: Regex,
    frequencies: HashMap<String, usize>,
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
            frequencies: HashMap::new(),
        }
    }

    /// Feed samples to the vocabulary generator. This will update the token
    /// frequency counts.
    pub fn feed(&mut self, samples: &[String]) {
        let frequencies = DashMap::new();
        let chunk_size = std::cmp::max(1, samples.len() / current_num_threads());

        samples.maybe_par_chunks(chunk_size).for_each(|chunk| {
            let thread_local_allow = self.allow.clone();
            let mut rng = rand::thread_rng();

            for sample in chunk {
                let mut sample_tokens = HashSet::new();
                for (i, _) in sample.char_indices() {
                    let suffix = &sample[i..];
                    for (ii, c) in suffix.char_indices().take(self.max_token_length) {
                        let candidate = &suffix[..ii + c.len_utf8()];

                        if thread_local_allow.is_match(candidate)
                            && rng.gen_range(0.0..1.0) < self.insert_probability
                        {
                            sample_tokens.insert(candidate);
                        }
                    }
                }

                for token in sample_tokens {
                    *frequencies.entry(token).or_insert(0) += 1;
                }
            }
        });

        for (token, freq) in frequencies.into_iter() {
            *self.frequencies.entry(token.into()).or_insert(0) += freq;
        }
    }

    /// Return the number of unique tokens in the vocabulary.
    pub fn current_size(&self) -> usize {
        self.frequencies.len()
    }

    /// Generate a vocabulary of the given size based on the frequency counts
    /// of the tokens fed to the generator. It returns the indices of the added
    /// tokens.
    pub fn generate(
        &mut self,
        size: usize,
        suggested_tokens: &HashSet<String>,
        added_tokens: &HashSet<String>,
    ) -> (Vec<ScoredToken>, HashSet<usize>) {
        let frequent_tokens = self
            .frequencies
            .clone()
            .into_iter()
            .collect::<HashMap<String, usize>>();

        // Collect the frequency of the added tokens.
        let added_tokens_freq = added_tokens
            .iter()
            .map(|token| frequent_tokens.get(token.as_str()).copied().unwrap_or(1))
            .collect::<Vec<usize>>();

        // Collect the frequency of the suggested tokens.
        let suggested_tokens_freq = suggested_tokens
            .iter()
            .map(|token| frequent_tokens.get(token.as_str()).copied().unwrap_or(1))
            .collect::<Vec<usize>>();

        // Convert the tokens to a vector and sort them by frequency.
        let mut frequent_tokens: Vec<_> = frequent_tokens.into_iter().collect();
        frequent_tokens.sort_by_key(|(_, freq)| Reverse(*freq));

        // Keep track of duplicates, ensuring the earlier occurrence is kept.
        let mut seen: HashSet<&str> = HashSet::new();
        let mut keep_indices = HashSet::new();

        // Add all 256 ASCII characters and byte values to the initial
        // vocabulary. We assume the frequency of each byte is the same as
        // the highest frequency token.
        let highest_freq = frequent_tokens.first().map(|(_, freq)| *freq).unwrap_or(1);
        let mut vocab: Vec<ScoredToken> = (0..255_u8)
            .map(|b| {
                keep_indices.insert(b as usize);
                (vec![b], highest_freq as f64)
            })
            .collect();

        // Add the added tokens.
        for (i, token) in added_tokens.iter().enumerate() {
            if !seen.contains(token.as_str()) && token.len() > 1 {
                seen.insert(token);
                vocab.push((
                    token.as_bytes().to_vec(),
                    (added_tokens_freq[i] as f64) * (token.len() as f64),
                ));
                keep_indices.insert(vocab.len() - 1);
            }
        }

        // Add the suggested tokens.
        for (i, token) in suggested_tokens.iter().enumerate() {
            if !seen.contains(token.as_str()) && token.len() > 1 {
                seen.insert(token);
                vocab.push((
                    token.as_bytes().to_vec(),
                    (suggested_tokens_freq[i] as f64) * (token.len() as f64),
                ));
            }
        }

        // We further add the most frequent substrings.
        for (token, freq) in &frequent_tokens {
            if vocab.len() >= size {
                break;
            }

            if !seen.contains(token.as_str()) && token.len() > 1 {
                seen.insert(token.as_str());
                vocab.push((token.as_bytes().to_vec(), (freq * token.len()) as f64));
            }
        }

        // Sort the vocabulary by score.
        vocab.sort_by(|(_, a), (_, b)| {
            a.partial_cmp(b)
                .unwrap_or(std::cmp::Ordering::Equal)
                .reverse()
        });

        // Convert the scores to log probabilities.
        logprobs(&mut vocab);

        // Computing log probabilities generates NaNs for items where freq=0.
        vocab.iter_mut().for_each(|(_, score)| {
            if !score.is_normal() {
                *score = 0.0;
            }
        });

        (vocab, keep_indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regexes() {
        let strict: u8 = 4;
        let base: u8 = 2;
        let advanced: u8 = 1;
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
