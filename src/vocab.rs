use crate::{logprobs, parallelism::*, ScoredToken, Token};
use dashmap::DashMap;
use rand::Rng;
use regex::Regex;
use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet},
};

pub const BASE_RE: &str = r#"(?:^.$)|(?:^ ?(?:[a-z]+):\/\/(?:(?:[a-z0-9]+\.?))+(?::[0-9]{1,5})?\/?$)|(?:^(?:(?:[\[{("']|__)[A-Za-z0-9]+(?:[\]})"']|__))$)|(?:^(?:[0-9]{1,4}(?:\.[0-9]{1,3})?)$)|(?:^(?:[\/\\_\-\.]|(?:__)|(?:::))[A-Za-z0-9_]+$)|(?:^(?:\/[a-zA-Z0-9]+)+$)|(?:^ ?[@&!?#$\*][A-Za-z0-9]+$)|(?:^ ?[A-Za-z0-9_]+(?:\(|%|(?:::)|(?:->)|(?:\.))$)|(?:^(?: ?(?:(?:[0-9]{1,4}(?:[A-Za-z]+)?)|(?:[A-Za-z]+(?:[0-9]{1,4})?)|[A-Za-z]+))$)|(?:^ ?(?:(?:[0-9]{1,4}(?:[a-zA-Z]+)?)|(?:[a-zA-Z]+(?:[0-9]{1,4})?)|[A-Z]+)(?:(?:[_\-.]|::|->)(?:(?:[0-9]{1,4}(?:[a-zA-Z]+)?)|(?:[a-zA-Z]+(?:[0-9]{1,4})?)|[A-Z]+))*$)|(?:^[\u3400-\u4DBF\u4E00-\u9FFF]+$)|(?:^(?:(?:(?:[[:punct:] DCU]+)?[[:space:]DCU]*)|(?:[[:space:]DCU]*(?:[[:punct:] DCU]+)?))$)|(?:^(?:<\/?[a-z]+(?:>|(?:\/>)|(?: \/>)|(?: @?[A-Za-z]+(?:=")?))?)$)|(?:^ ?(?:(?:[A-Za-z]+(?:'[A-Za-z]+)?)|(?:[0-9]{1,4}))(?:(?: [A-Za-z]+(?:'[A-Za-z]+)?)|(?: [0-9]{1,4})){0,2}$)|(?:^ ?(?:(?:!=)|(?:!==)|(?:==)|(?:=)) ?[A-Za-z0-9_]+(?:\.[0-9]{1,3})?$)|(?:^(?:, ?[A-Za-z0-9]+)$)"#;
pub const CAPCODE_RE: &str = r#"(?:^.$)|(?:^ (?:[a-z]+):\/\/(?:(?:D? [a-z0-9]+(?:\.D)?))+(?::D [0-9]{1,5})?(?:\/D)?$)|(?:^(?:[\[{("']D[UC]? [a-z0-9]+[\]})"'])$)|(?:^(?:D?[UC]? ?[0-9]{1,4}(?:\.D [0-9]{1,2})?)$)|(?:^[\/\\_\-\.]D[UC]? (?:(?:[a-z0-9]+)|(?:D[UC]? ))+$)|(?:^ ?(?:[\[(@&!?#$\*]|::)D[UC]? [a-z0-9]+$)|(?:^ ?[a-z0-9]+(?:%|(?:\(D[UC]?))$)|(?:^D?[UC]? ?(?:(?:(?:[0-9]{1,4})(?:D[UC]? (?:[a-z]+))?)|(?:(?:[a-z]+)(?:D[UC]? (?:[0-9]{1,4}))?)|(?:[a-z]+)(D[UC]? [a-z]+)*)$)|(?:^[UC]? ?(?:(?:(?:[0-9]{1,4})(?:D[UC]? (?:[a-z]+))?)|(?:(?:[a-z]+)(?:D[UC]? (?:[0-9]{1,4}))?)|(?:[a-z]+)(D[UC]? [a-z]+)*)(?:(?:[_\-.]|::)D[UC]? (?:(?:(?:[0-9]{1,4})(?:D[UC]? (?:[a-z]+))?)|(?:(?:[a-z]+)(?:D[UC]? (?:[0-9]{1,4}))?)|(?:[a-z]+)(D[UC]? [a-z]+)*))*$)|(?:^[\u3400-\u4DBF\u4E00-\u9FFF]+$)|(?:^(?:(?:(?:[[:punct:] DCU]+)?[[:space:]DCU]*)|(?:[[:space:]DCU]*(?:[[:punct:] DCU]+)?))$)|(?:^(?:<\/?D[UC]? [a-z]+(?:>|(?:\/>)|(?: \/>)|(?: (?:@D )?[a-z]+(?:="D)?))?)$)|(?:^(?:[UC]?(?:(?: [a-z]+(?:'[a-z]+)?))|(?: [0-9]{1,4})){1,3}$)|(?:^(?:,D?[UC]? ?[a-z0-9]+)$)|(?:^(?:\/D [a-zA-Z0-9]+)+$)|(?:^ ?(?:(?:!=)|(?:!==)|(?:==)|(?:=))D?[UC]? (?:(?:[a-z]+)|(?:[0-9]{1,3}(?:\.[0-9]{1,3})?))$)"#;

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
            let mut sample_tokens = HashSet::new();

            for sample in chunk {
                for (i, _) in sample.char_indices() {
                    let mut len = 0;
                    let suffix = &sample[i..];

                    for (ii, c) in suffix.char_indices() {
                        len += c.len_utf8();

                        if len > self.max_token_length {
                            break;
                        }

                        let candidate = &suffix[..ii + c.len_utf8()];

                        if thread_local_allow.is_match(candidate)
                            && rng.gen_range(0.0..1.0) < self.insert_probability
                        {
                            sample_tokens.insert(candidate);
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
    /// of the tokens fed to the generator. It returns the indices of the added
    /// tokens.
    pub fn generate(
        &mut self,
        size: usize,
        suggested_tokens: &HashSet<String>,
        added_tokens: &HashSet<String>,
    ) -> (Vec<ScoredToken>, HashSet<Token>) {
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
        frequent_tokens.sort_unstable_by_key(|(_, freq)| Reverse(*freq));

        // Keep track of duplicates, ensuring the earlier occurrence is kept.
        let mut seen: HashSet<&str> = HashSet::new();
        // Record which tokens cannot be pruned.
        let mut keep = HashSet::<Token>::new();

        // Add all 256 ASCII characters and byte values to the initial
        // vocabulary. We assume the frequency of each byte is the same as
        // the highest frequency token.
        let highest_freq = frequent_tokens.first().map(|(_, freq)| *freq).unwrap_or(1);
        let mut vocab: Vec<ScoredToken> = (0..255_u8)
            .map(|b| {
                keep.insert(vec![b]);
                (vec![b], highest_freq as f64)
            })
            .collect();

        // Add the added tokens.
        for (i, token) in added_tokens.iter().enumerate() {
            if !seen.contains(token.as_str()) && token.len() > 1 {
                seen.insert(token);
                keep.insert(token.as_bytes().to_vec());
                vocab.push((
                    token.as_bytes().to_vec(),
                    (added_tokens_freq[i] * token.len()) as f64,
                ));
            }
        }

        // Add the suggested tokens.
        for (i, token) in suggested_tokens.iter().enumerate() {
            if !seen.contains(token.as_str()) && token.len() > 1 {
                seen.insert(token);
                vocab.push((
                    token.as_bytes().to_vec(),
                    (suggested_tokens_freq[i] * token.len()) as f64,
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
        vocab.sort_unstable_by(|(_, a), (_, b)| {
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

        (vocab, keep)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_regex_matches(re: &Regex, ok: &[&str], nok: &[&str]) {
        for sample in ok {
            assert!(re.is_match(sample), "Expected {:?} to match {}", sample, re);
        }

        for sample in nok {
            assert!(
                !re.is_match(sample),
                "Expected {:?} not to match {}",
                sample,
                re
            );
        }
    }

    fn spaced_url_capcode_regex() -> Regex {
        let protocol = r#"(?:[a-z]+)"#;
        let subdomains = r#"(?:(?:D? [a-z0-9]+(?:\.D)?))+"#;
        let port = r#"(?::D [0-9]{1,5})"#;
        let trailing_slash = r#"(?:\/D)"#;

        let re = format!(
            r#"^ {}:\/\/{}{}?{}?$"#,
            protocol, subdomains, port, trailing_slash
        );
        let re = Regex::new(&re).unwrap();

        assert_regex_matches(
            &re,
            &[
                " http://D www.D google.D com",
                " http://D www.D google.D com/D",
                " http://D www.D google.D com:D 8443",
                " http://D www.D google.D com:D 8443/D",
                " https://D 127.D 0.D 0.D 1",
                " https://D 127.D 0.D 0.D 1:D 80",
                " https://D 127.D 0.D 0.D 1:D 80/D",
                " https://D www.D",
                " https://D www",
                " tcp://D github.D com",
            ],
            &["https://D github.D com", "https"],
        );

        re
    }

    fn wrapped_in_punctuation_capcode_regex() -> Regex {
        let pattern = r#"(?:[\[{("']D[UC]? [a-z0-9]+[\]})"'])"#;

        let re = format!("^{}$", pattern);
        let re = Regex::new(&re).unwrap();

        assert_regex_matches(
            &re,
            &[
                "[D word]",
                "[D word]",
                "[D 1]",
                "(D word)",
                "(D 123)",
                "{D word}",
                "{D 123}",
                "[DC word]",
                "[DU word]",
                "'D word'",
                "'D 123'",
                "\"D word\"",
                "\"D 123\"",
            ],
            &["[D wordD", "D wordD]", "D wordD", "D word"],
        );

        re
    }

    fn number_capcode_regex() -> Regex {
        let number = r#"(?:D?[UC]? ?[0-9]{1,4}(?:\.D [0-9]{1,2})?)"#;

        let re = format!("^{}$", number);
        let re = Regex::new(&re).unwrap();

        assert_regex_matches(
            &re,
            &["1", "123", "1234", "1000.D 0", " 50.D 1"],
            &["12345", " 12345", " 0.D 123"],
        );

        re
    }

    fn punctuation_word_capcode_regex() -> Regex {
        let pattern = r#"[\/\\_\-\.]D[UC]? (?:(?:[a-z0-9]+)|(?:D[UC]? ))+"#;

        let re = format!("^{}$", pattern);
        let re = Regex::new(&re).unwrap();

        assert_regex_matches(
            &re,
            &[".DU word", "/D 123", "\\D word", ".DC toDC string"],
            &[" /D word"],
        );

        re
    }

    fn space_punctuation_word_capcode_regex() -> Regex {
        let pattern = r#" ?(?:[\[(@&!?#$\*]|::)D[UC]? [a-z0-9]+"#;

        let re = format!("^{}$", pattern);
        let re = Regex::new(&re).unwrap();

        assert_regex_matches(&re, &[" &DU word", "@D 123", " !D word"], &[":D word"]);

        re
    }

    fn space_word_punctuation_capcode_regex() -> Regex {
        let pattern = r#" ?[a-z0-9]+(?:%|(?:\(D[UC]?))"#;

        let re = format!("^{}$", pattern);
        let re = Regex::new(&re).unwrap();

        assert_regex_matches(&re, &[" 10%", " 100%"], &[]);

        re
    }

    fn camel_case_capcode_regex() -> Regex {
        let word_and_number =
            r#"(?:(?:[0-9]{1,4})(?:D[UC]? (?:[a-z]+))?)|(?:(?:[a-z]+)(?:D[UC]? (?:[0-9]{1,4}))?)"#;
        let words = r#"(?:[a-z]+)(D[UC]? [a-z]+)*"#;
        let camel_case = format!("D?[UC]? ?(?:{}|{})", word_and_number, words);

        let re = format!("^{}$", camel_case);
        let re = Regex::new(&re).unwrap();

        assert_regex_matches(
            &re,
            &[
                "uD 32",
                " iD 32",
                "123D word",
                "D word",
                "DU word",
                "DC 123D word",
                "wordDC wordDC word",
            ],
            &["12345", " 12345", " wordD 12345"],
        );

        re
    }

    fn snake_case_namespaces_capcode_regex() -> Regex {
        let word_and_number =
            r#"(?:(?:[0-9]{1,4})(?:D[UC]? (?:[a-z]+))?)|(?:(?:[a-z]+)(?:D[UC]? (?:[0-9]{1,4}))?)"#;
        let words = r#"(?:[a-z]+)(D[UC]? [a-z]+)*"#;
        let snake_case = format!(
            "[UC]? ?(?:{}|{})(?:(?:[_\\-.]|::)D[UC]? (?:{}|{}))*",
            word_and_number, words, word_and_number, words
        );

        let re = format!("^{}$", snake_case);
        let re = Regex::new(&re).unwrap();

        assert_regex_matches(
            &re,
            &[
                " my_D var",
                " my_D 123",
                " uint_D 32_D t",
                " std::D var",
                " name.D space",
                " name-D space",
            ],
            &["12345", " 12345", " wordD 12345"],
        );

        re
    }

    fn chinese_regex() -> Regex {
        let re = Regex::new(r#"^[\u3400-\u4DBF\u4E00-\u9FFF]+$"#).unwrap();

        assert_regex_matches(
            &re,
            &["好", "你好", "你好吗", "你好吗", "你好吗好"],
            &["好 了"],
        );

        re
    }

    fn punctuation_whitespace() -> Regex {
        // Punctuation with space.
        let many_punct_capcode_or_space = r#"(?:[[:punct:] DCU]+)"#;
        // L punctuation with whitespace.
        let l_punct_whitespace = format!("(?:{}?[[:space:]DCU]*)", many_punct_capcode_or_space);
        // R punctuation with whitespace.
        let r_punct_whitespace = format!("(?:[[:space:]DCU]*{}?)", many_punct_capcode_or_space);

        let re = format!("^(?:{}|{})$", l_punct_whitespace, r_punct_whitespace);
        let re = Regex::new(&re).unwrap();

        assert_regex_matches(
            &re,
            &[
                " ",
                " . ",
                ".",
                " .",
                ". / ^ \t\n",
                " \t \n . / ^ ",
                "}\n\n\tD",
            ],
            &["\na\n", "a\na", "\n:\n"],
        );

        re
    }

    fn html_tag_capcode_regex() -> Regex {
        let html_tag =
            r#"(?:<\/?D[UC]? [a-z]+(?:>|(?:\/>)|(?: \/>)|(?: (?:@D )?[a-z]+(?:="D)?))?)"#;

        let re = format!("^{}$", html_tag);
        let re = Regex::new(&re).unwrap();
        assert_regex_matches(
            &re,
            &[
                "<D div>",
                "</D div>",
                "<DU div>",
                "<D div @D click=\"D",
                "<D div class=\"D",
                "<D div class",
                "<DC div>",
                "<D img/>",
                "<D img />",
            ],
            &["a>", "<D a /", "<D a / >"],
        );

        re
    }

    fn multiple_words_capcode_regex() -> Regex {
        let pattern = r#"(?:[UC]?(?:(?: [a-z]+(?:'[a-z]+)?))|(?: [0-9]{1,4})){1,3}"#;

        let re = format!("^{}$", pattern);
        let re = Regex::new(&re).unwrap();

        assert_regex_matches(
            &re,
            &[
                " multiple words",
                " don't do",
                " word",
                " my 123 word",
                "U warrantyU whatever",
            ],
            &["not", "not that", "123 not"],
        );

        re
    }

    fn equal_word_capcode_regex() -> Regex {
        let pattern = r#" ?(?:(?:!=)|(?:!==)|(?:==)|(?:=))D?[UC]? (?:(?:[a-z]+)|(?:[0-9]{1,3}(?:\.[0-9]{1,3})?))"#;

        let re = format!("^{}$", pattern);
        let re = Regex::new(&re).unwrap();

        assert_regex_matches(
            &re,
            &[
                " == nil",
                "!== undefined",
                " = 1.0",
                "= value",
                " =D value",
                "=D value",
            ],
            &["12345"],
        );

        re
    }

    fn paths_capcode_regex() -> Regex {
        let path = r#"(?:\/D [a-zA-Z0-9]+)+"#;

        let re = format!("^{}$", path);
        let re = Regex::new(&re).unwrap();

        assert_regex_matches(
            &re,
            &[
                "/D hello/D 123",
                "/D hello/D 123/D 456",
                "/D hello/D 123/D 456/D 789",
                "/D hello/D 123/D 456/D 789/D abc",
                "/D hello/D 123/D 456/D 789/D abc/D def",
            ],
            &[],
        );

        re
    }

    fn comma_word_capcode_regex() -> Regex {
        let pattern = r#"(?:,D?[UC]? ?[a-z0-9]+)"#;

        let re = format!("^{}$", pattern);
        let re = Regex::new(&re).unwrap();

        assert_regex_matches(
            &re,
            &[",D word", ",U word", ", 123"],
            &["12345", " 12345", " word12345"],
        );

        re
    }

    #[test]
    fn test_capcode_regexes() {
        // Single character
        let lone_char = Regex::new(r#"^.$"#).unwrap();
        // Space + URL
        let spaced_url = spaced_url_capcode_regex();
        // Wrapped in punctuation
        let wrapped_in_punctuation = wrapped_in_punctuation_capcode_regex();
        // Number and decimal number
        let number = number_capcode_regex();
        // Punctuation + word
        let punctuation_word = punctuation_word_capcode_regex();
        // Space + punctuation + word
        let space_punctuation_word = space_punctuation_word_capcode_regex();
        // Space + word + punctuation
        let space_word_punctuation = space_word_punctuation_capcode_regex();
        // Camel case
        let camel_case = camel_case_capcode_regex();
        // Snake case
        let snake_case = snake_case_namespaces_capcode_regex();
        // Chinese
        let chinese = chinese_regex();
        // Punctuation with whitespace
        let punctuation_whitespace = punctuation_whitespace();
        // HTML tag
        let html_tag = html_tag_capcode_regex();
        // Multiple words
        let multiple_words = multiple_words_capcode_regex();
        // Equal word
        let equal_word = equal_word_capcode_regex();
        // Paths
        let paths = paths_capcode_regex();
        // Comma
        let comma_word = comma_word_capcode_regex();

        let capcode_regex = Regex::new(
            &[
                lone_char,
                spaced_url,
                wrapped_in_punctuation,
                number,
                punctuation_word,
                space_punctuation_word,
                space_word_punctuation,
                camel_case,
                snake_case,
                chinese,
                punctuation_whitespace,
                html_tag,
                multiple_words,
                comma_word,
                paths,
                equal_word,
            ]
            .map(|re| format!("(?:{})", re.as_str()))
            .join("|"),
        )
        .unwrap();

        assert_regex_matches(
            &capcode_regex,
            &[
                // Words
                "a",
                "abc",
                // Numbers
                "1",
                "123",
                " 123",
                " 1000.D 0",
                // Whitespace
                "\n\n\n",
                "    ",
                // Spaced URL
                " https://D word",
                " https://D word.D com",
                // Wrapped in punctuation
                "(D word)",
                "[D 1]",
                // Paths
                // "/D word/D word",
                // Prefix Punctuation
                "/D word",
                ".D word",
                "*D word",
                " *D word",
                "@DU word",
                " &D word",
                " 10%",
                // Camel case
                "DC myDC var",
                "myDC varDC myDC var",
                " 123DC word",
                " wordD 123",
                // Snake case
                " word_D word",
                " wordD 123_D word",
                "U word_DU word",
                // Multi-word
                " in order to",
                " don't mind the",
                // Chinese
                "好",
                "你好",
                // Punctuation with whitespace
                "\n\n} DCU",
                "}\n\nDCU",
                // Some more
                "::D new",
                "[D i",
                "(D self",
                "!= nil",
                "=DC none",
                "print(D",
                "/D usr/D bin",
                ", 0",
                "::",
            ],
            &[
                "12345",
                " 12345",
                " wordD 12345",
                " /D word",
                "\na\n",
                "a\na",
                "D 033[D 0",
            ],
        );

        println!("{}", capcode_regex);
    }

    fn maybe_spaced_url_regex() -> Regex {
        let protocol = r#"(?:[a-z]+)"#;
        let subdomains = r#"(?:(?:[a-z0-9]+\.?))+"#;
        let port = r#"(?::[0-9]{1,5})"#;

        let re = format!(r#"^ ?{}:\/\/{}{}?\/?$"#, protocol, subdomains, port);
        let re = Regex::new(&re).unwrap();

        assert_regex_matches(
            &re,
            &[
                " http://www.google.com",
                "http://www.google.com/",
                " http://www.google.com:8443",
                "http://www.google.com:8443/",
                " https://127.0.0.1",
                "https://127.0.0.1:80",
                " https://127.0.0.1:80/",
                "https://www.",
                " https://www",
                "tcp://github.com",
            ],
            &[],
        );

        re
    }

    fn wrapped_in_punctuation_regex() -> Regex {
        let pattern = r#"(?:(?:[\[{("']|__)[A-Za-z0-9]+(?:[\]})"']|__))"#;

        let re = format!("^{}$", pattern);
        let re = Regex::new(&re).unwrap();

        assert_regex_matches(
            &re,
            &[
                "[word]",
                "[word]",
                "[1]",
                "(word)",
                "(123)",
                "{word}",
                "{123}",
                "[WORD]",
                "__device__",
                "[Word]",
                "'word'",
                "'123'",
                "\"word\"",
                "\"123\"",
            ],
            &[],
        );

        re
    }

    fn number_regex() -> Regex {
        let number = r#"(?:[0-9]{1,4}(?:\.[0-9]{1,3})?)"#;

        let re = format!("^{}$", number);
        let re = Regex::new(&re).unwrap();

        assert_regex_matches(
            &re,
            &["1", "123", "1234", "1000.0", "50.1"],
            &["12345", " 12345", " 0.123"],
        );

        re
    }

    fn punctuation_word_regex() -> Regex {
        let pattern = r#"(?:[\/\\_\-\.]|(?:__)|(?:::))[A-Za-z0-9_]+"#;

        let re = format!("^{}$", pattern);
        let re = Regex::new(&re).unwrap();

        assert_regex_matches(
            &re,
            &[".word", "/123", "\\word", "__var", "_THIS_THING"],
            &[],
        );

        re
    }

    fn space_punctuation_word_regex() -> Regex {
        let pattern = r#" ?[@&!?#$\*][A-Za-z0-9]+"#;

        let re = format!("^{}$", pattern);
        let re = Regex::new(&re).unwrap();

        assert_regex_matches(&re, &[" &word", "@123", " !word", "*t"], &[":word"]);

        re
    }

    fn space_word_punctuation_regex() -> Regex {
        let pattern = r#" ?[A-Za-z0-9_]+(?:\(|%|(?:::)|(?:->)|(?:\.))"#;

        let re = format!("^{}$", pattern);
        let re = Regex::new(&re).unwrap();

        assert_regex_matches(&re, &[" 10%", " 100%", "std::"], &[]);

        re
    }

    fn camel_case_regex() -> Regex {
        let word_and_number = r#"(?:[0-9]{1,4}(?:[A-Za-z]+)?)|(?:[A-Za-z]+(?:[0-9]{1,4})?)"#;
        let words = r#"[A-Za-z]+"#;
        let camel_case = format!("(?: ?(?:{}|{}))", word_and_number, words);

        let re = format!("^{}$", camel_case);
        let re = Regex::new(&re).unwrap();

        assert_regex_matches(
            &re,
            &[
                "32",
                " i32",
                "123word",
                "word",
                " WORDword",
                "123word",
                " wordWordWord",
            ],
            &["12345", " 12345", " word12345"],
        );

        re
    }

    fn snake_case_namespaces_regex() -> Regex {
        let word_and_number = r#"(?:[0-9]{1,4}(?:[a-zA-Z]+)?)|(?:[a-zA-Z]+(?:[0-9]{1,4})?)"#;
        let words = r#"[A-Z]+"#;
        let snake_case = format!(
            " ?(?:{}|{})(?:(?:[_\\-.]|::|->)(?:{}|{}))*",
            word_and_number, words, word_and_number, words
        );

        let re = format!("^{}$", snake_case);
        let re = Regex::new(&re).unwrap();

        assert_regex_matches(
            &re,
            &[
                "my_var",
                " my_123",
                " uint_32_t",
                "std.var",
                " name.space",
                " name::space",
                " hello->world",
                " hello-world",
            ],
            &["12345", " 12345", " word12345"],
        );

        re
    }

    fn html_tag_regex() -> Regex {
        let html_tag = r#"(?:<\/?[a-z]+(?:>|(?:\/>)|(?: \/>)|(?: @?[A-Za-z]+(?:=")?))?)"#;

        let re = format!("^{}$", html_tag);
        let re = Regex::new(&re).unwrap();
        assert_regex_matches(
            &re,
            &[
                "<div>", "</div>", "<div>", "<div>", "<img/>", "<img />", "<div",
            ],
            &["a>", "<a /", "<a / >"],
        );

        re
    }

    fn multiple_words_regex() -> Regex {
        let pattern = r#" ?(?:(?:[A-Za-z]+(?:'[A-Za-z]+)?)|(?:[0-9]{1,4}))(?:(?: [A-Za-z]+(?:'[A-Za-z]+)?)|(?: [0-9]{1,4})){0,2}"#;

        let re = format!("^{}$", pattern);
        let re = Regex::new(&re).unwrap();

        assert_regex_matches(
            &re,
            &[
                " multiple words",
                " 2 words",
                " don't do",
                "word",
                " word",
                " my 123 word",
                "WARRANTY WHATEVER",
            ],
            &["four words or more"],
        );

        re
    }

    fn paths_regex() -> Regex {
        let path = r#"(?:\/[a-zA-Z0-9]+)+"#;

        let re = format!("^{}$", path);
        let re = Regex::new(&re).unwrap();

        assert_regex_matches(
            &re,
            &[
                "/hello/123",
                "/hello/123/456",
                "/hello/123/456/789",
                "/hello/123/456/789/abc",
                "/hello/123/456/789/abc/def",
            ],
            &[],
        );

        re
    }

    fn equal_word_regex() -> Regex {
        let pattern = r#" ?(?:(?:!=)|(?:!==)|(?:==)|(?:=)) ?[A-Za-z0-9_]+(?:\.[0-9]{1,3})?"#;

        let re = format!("^{}$", pattern);
        let re = Regex::new(&re).unwrap();

        assert_regex_matches(
            &re,
            &[
                " == nil",
                "!== undefined",
                " = 1.0",
                "= value",
                " = value",
                "= value",
            ],
            &["12345"],
        );

        re
    }

    fn comma_word_regex() -> Regex {
        let pattern = r#"(?:, ?[A-Za-z0-9]+)"#;

        let re = format!("^{}$", pattern);
        let re = Regex::new(&re).unwrap();

        assert_regex_matches(
            &re,
            &[",word", ", word", ", 123"],
            &["12345", " 12345", " word12345"],
        );

        re
    }

    #[test]
    fn test_base_regexes() {
        // Single character
        let lone_char = Regex::new(r#"^.$"#).unwrap();
        // Maybe Space + URL
        let maybe_spaced_url = maybe_spaced_url_regex();
        // Wrapped in punctuation
        let wrapped_in_punctuation = wrapped_in_punctuation_regex();
        // Number and decimal number
        let number = number_regex();
        // Punctuation + word
        let punctuation_word = punctuation_word_regex();
        // Space + punctuation + word
        let space_punctuation_word = space_punctuation_word_regex();
        // Space + word + punctuation
        let space_word_punctuation = space_word_punctuation_regex();
        // Camel case
        let camel_case = camel_case_regex();
        // Snake case
        let snake_case = snake_case_namespaces_regex();
        // Paths
        let paths = paths_regex();
        // Chinese
        let chinese = chinese_regex();
        // Punctuation with whitespace
        let punctuation_whitespace = punctuation_whitespace();
        // HTML tag
        let html_tag = html_tag_regex();
        // Multiple words
        let multiple_words = multiple_words_regex();
        // Equal
        let equal = equal_word_regex();
        // Comma
        let comma = comma_word_regex();

        let base_regex = Regex::new(
            &[
                lone_char,
                maybe_spaced_url,
                wrapped_in_punctuation,
                number,
                punctuation_word,
                paths,
                space_punctuation_word,
                space_word_punctuation,
                camel_case,
                snake_case,
                chinese,
                punctuation_whitespace,
                html_tag,
                multiple_words,
                equal,
                comma,
            ]
            .map(|re| format!("(?:{})", re.as_str()))
            .join("|"),
        )
        .unwrap();
        assert_regex_matches(
            &base_regex,
            &[
                // Words
                "a",
                "abc",
                " word",
                // Numbers
                "1",
                "123",
                " 456",
                "123.456",
                " 1000.0",
                // Whitespace
                "\n\n\n",
                "    ",
                // Maybe Spaced URL
                " http://www.google.com",
                "http://github.com:8443/",
                // Wrapped in punctuation
                "(word)",
                "[1]",
                "{WORD}",
                // Prefix Punctuation
                "/word",
                ".word",
                "*word",
                " *word",
                "@word",
                " &word",
                " 10%",
                // Camel case
                "myVar",
                "my123",
                "ManyWordsTOGETHER",
                // Snake case
                "word_word",
                "word_123_word",
                "MANY_WORDS",
                // Namespaces
                "/hello/123",
                "System.IO",
                "std::var",
                // Multi-word
                "multiple words",
                "don't do",
                " in order to",
                // Chinese
                "好",
                "你好",
                // Punctuation with whitespace
                "\n\n}",
                "}\n\n",
                // HTML tag
                "<div>",
                "<div @click=\"",
                "<div class=\"",
                "</div>",
                "<img/>",
                "<img />",
                "<div",
                " . ",
                // Special
                "::is",
                "::is_special",
                "std::",
                "__forceinline__",
                "starts_with(",
                "@TEMPLATE",
                "#include",
                " == nil",
                " = 0",
                " == True",
                "'hello'",
                "(error)",
                "pointer->",
                "this.",
            ],
            &[
                // Too long numbers
                "12345", " 12345",
            ],
        );
        println!("{}", base_regex);
    }
}
