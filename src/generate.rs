use dashmap::DashMap;
use fancy_regex::Regex as FancyRegex;
use rand::Rng;
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use regex::Regex;
use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet},
};
use tokengeex::{par_chunk_size, ScoredToken, Token};

pub struct VocabularyGenerator {
    max_token_length: usize,
    insert_probability: f64,
    split: Option<FancyRegex>,
    allow: Option<Regex>,
    added_tokens: Vec<String>,
    suggested_tokens: Vec<String>,
    frequencies: HashMap<String, usize>,
}

impl VocabularyGenerator {
    pub fn new(
        max_token_length: usize,
        insert_probability: f64,
        split: Option<FancyRegex>,
        allow: Option<Regex>,
        added_tokens: Vec<String>,
        suggested_tokens: Vec<String>,
    ) -> VocabularyGenerator {
        let mut frequencies = HashMap::new();

        for token in &added_tokens {
            *frequencies.entry(token.clone()).or_insert(0) += 1;
        }

        for token in &suggested_tokens {
            *frequencies.entry(token.clone()).or_insert(0) += 1;
        }

        VocabularyGenerator {
            max_token_length,
            insert_probability,
            split,
            allow,
            added_tokens,
            suggested_tokens,
            frequencies,
        }
    }

    /// Feed samples to the vocabulary generator. This will update the token
    /// frequency counts.
    pub fn feed(&mut self, samples: &[String]) {
        let frequencies = DashMap::new();
        let chunk_size = par_chunk_size(samples.len(), 5);

        samples.par_chunks(chunk_size).for_each(|chunk| {
            let thread_local_allow = self.allow.clone();
            let thread_local_split = self.split.clone();
            let mut rng = rand::thread_rng();
            let mut sample_tokens = HashSet::new();

            for sample in chunk {
                if let Some(split) = &thread_local_split {
                    for part in split.find_iter(sample) {
                        let part = part.unwrap().as_str();

                        for (i, _) in part.char_indices() {
                            let mut len = 0;
                            let suffix = &part[i..];

                            for (ii, c) in suffix.char_indices() {
                                len += c.len_utf8();

                                if len > self.max_token_length {
                                    break;
                                }

                                let candidate = &suffix[..ii + c.len_utf8()];

                                if thread_local_allow
                                    .as_ref()
                                    .map_or(true, |allow| allow.is_match(candidate))
                                    && rng.gen_range(0.0..1.0) < self.insert_probability
                                {
                                    sample_tokens.insert(candidate);
                                }
                            }
                        }
                    }
                } else {
                    for (i, _) in sample.char_indices() {
                        let mut len = 0;
                        let suffix = &sample[i..];

                        for (ii, c) in suffix.char_indices() {
                            len += c.len_utf8();

                            if len > self.max_token_length {
                                break;
                            }

                            let candidate = &suffix[..ii + c.len_utf8()];

                            if thread_local_allow
                                .as_ref()
                                .map_or(true, |allow| allow.is_match(candidate))
                                && rng.gen_range(0.0..1.0) < self.insert_probability
                            {
                                sample_tokens.insert(candidate);
                            }
                        }
                    }
                }

                // Find all occurences of added and suggested tokens in the
                // sample and add them to the frequencies.
                for token in self.added_tokens.iter().chain(self.suggested_tokens.iter()) {
                    for _ in sample.match_indices(token) {
                        if rng.gen_range(0.0..1.0) < self.insert_probability {
                            sample_tokens.insert(token);
                            break;
                        }
                    }
                }

                for &token in &sample_tokens {
                    *frequencies.entry(token).or_insert(0) += 1;
                }

                sample_tokens.clear();
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
    /// of the tokens fed to the generator.
    pub fn generate(&mut self, size: usize) -> Vec<ScoredToken> {
        // Convert the tokens to a vector and sort them by frequency.
        let mut frequent_tokens: Vec<_> = self.frequencies.iter().collect();
        frequent_tokens.sort_unstable_by_key(|(_, freq)| Reverse(*freq));

        // Keep track of duplicates, ensuring the earlier occurrence is kept.
        let mut seen: HashSet<Token> = HashSet::new();

        // Add all 256 ASCII characters and byte values to the initial
        // vocabulary. We assume the frequency of each byte is the same as
        // the highest frequency token.
        let highest_freq = frequent_tokens
            .first()
            .map(|(_, freq)| *freq)
            .copied()
            .unwrap_or(1);
        let mut vocab: Vec<ScoredToken> = (0..255_u8)
            .map(|b| {
                seen.insert(vec![b]);
                ScoredToken::from_u8(b, highest_freq as f64, true)
            })
            .collect();

        // We add the suggested and added tokens.
        for (token, keep) in self
            .added_tokens
            .iter()
            .map(|v| (v, true))
            .chain(self.suggested_tokens.iter().map(|v| (v, false)))
        {
            if vocab.len() >= size {
                break;
            }

            if !seen.contains(&token.as_bytes().to_vec()) && token.len() > 1 {
                seen.insert(token.as_bytes().to_vec());

                let score = self
                    .frequencies
                    .get(token)
                    .map(|freq| (*freq * token.len()) as f64)
                    .expect("suggested/added token score should be present");

                vocab.push(ScoredToken::from_str(token, score, keep));
            }
        }

        // We further add the most frequent substrings.
        for (token, freq) in &frequent_tokens {
            if vocab.len() >= size {
                break;
            }

            if !seen.contains(&token.as_bytes().to_vec()) && token.len() > 1 {
                seen.insert(token.as_bytes().to_vec());
                vocab.push(ScoredToken::from_str(
                    token,
                    (*freq * token.len()) as f64,
                    false,
                ));
            }
        }

        // Sort the vocabulary by score.
        vocab.sort_unstable_by(|a, b| {
            a.partial_cmp(b)
                .unwrap_or(std::cmp::Ordering::Equal)
                .reverse()
        });

        // Convert the scores to log probabilities.
        logprobs(&mut vocab);

        // Computing log probabilities generates NaNs for items where freq=0.
        vocab.iter_mut().for_each(|token| {
            if !token.score.is_normal() {
                panic!(
                    "Vocabulary generation: invalid frequency for token {:?}, {:?}: {}",
                    token.value,
                    String::from_utf8_lossy(&token.value),
                    token.score
                );
            }
        });

        vocab
    }
}

pub fn logprobs(pieces: &mut [ScoredToken]) {
    let sum: f64 = pieces.iter().map(|token| token.score).sum();
    let logsum = sum.ln();
    for token in pieces.iter_mut() {
        token.score = token.score.ln() - logsum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use regex::Regex;

    #[test]
    fn test_generate() {
        let mut generator = VocabularyGenerator::new(
            6,
            1.0,
            None,
            Some(Regex::new(r#"^ ?[a-z]+$"#).unwrap()),
            vec!["goodbye".into(), "vec".into()],
            vec!["string".into(), "map".into()],
        );

        let samples = vec![
            "hello my name is diego and i like std::string".into(),
            "i also like std::vector".into(),
            "and std::vector<std::string>".into(),
            "and std::map<int, std::string>".into(),
        ];

        generator.feed(&samples);
        let mut vocab = generator.generate(256 + 10);

        // Drop all byte-level tokens.
        vocab.retain(|token| token.value.len() > 1);

        println!("vocab: {:#?}", vocab);

        let expected = vec!["string"];
        for token in expected {
            assert!(
                vocab.iter().any(|t| t.value == token.as_bytes()),
                "missing token: {}",
                token
            );
        }
    }
}
