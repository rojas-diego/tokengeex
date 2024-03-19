use crate::{logprobs, parallelism::*, ScoredToken, Token};
use dashmap::DashMap;
use rand::Rng;
use regex::Regex;
use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet},
};

pub const STRICT_RE: &str = r#"(?:^.$)|(?:^(?:(?:[[:punct:]]|(?:::))(?:(?:DU|DC|D) )(?:[a-z0-9]+))$)|(?:^(?:(?:[[:punct:] DCU]+)?(?:[[:space:]]*))$)|(?:^(?:[[:space:]]*(?:[[:punct:] DCU]+)?)$)|(?:^(?:^[\u3400-\u4DBF\u4E00-\u9FFF]+)$)|(?:^(?: (?:[a-z]+)://(?:(?:(?:(?:(?:(?:DU|DC|D) )(?:[a-z]+))(?:-(?:(?:(?:DU|DC|D) )(?:[a-z]+)))*)(?:\.(?:(?:(?:(?:DU|DC|D) )(?:[a-z]+))(?:-(?:(?:(?:DU|DC|D) )(?:[a-z]+)))*))*)|(?:(?:(?:DU|DC|D) )?(?:[0-9]+)(?:\.(?:(?:(?:DU|DC|D) )(?:[0-9]+))){3}))(?::(?:(?:DU|DC|D) )[0-9]{1,5})?)$)|(?:^(?:<D?[UC]? [a-z]+(?:>|/>| />)?)$)|(?:^(?:(?:(?:(?:(?:D|DU|DC|U|C) )| )?(?:[0-9]+))|(?:(?:(?:(?:D|DU|DC|U|C) )| )?(?:[a-z]+))){1,3}$)"#;
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
                    (added_tokens_freq[i] as f64) * (token.len() as f64),
                ));
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
            assert!(
                re.is_match(sample),
                "Expected {:?} to match {:?}",
                sample,
                re
            );
        }

        for sample in nok {
            assert!(
                !re.is_match(sample),
                "Expected {:?} not to match {:?}",
                sample,
                re
            );
        }
    }

    #[test]
    fn make_regexes_capcode() {
        // ---------------------------------------------------------------------
        // Basic
        // ---------------------------------------------------------------------
        // Capcode marker sequence including the space.
        let capcode = r#"(?:(?:D|DU|DC|U|C) )"#;
        // Capcode marker sequence including the space and the D marker.
        let delete_capcode = r#"(?:(?:DU|DC|D) )"#;
        // Any lowercase word.
        let word = r#"(?:[a-z]+)"#;
        // Any number.
        let number = r#"(?:[0-9]+)"#;
        // Small number (max 3 digits).
        let _small_number = r#"(?:[0-9]{1,3})"#;
        // Word or number.
        let word_or_number = r#"(?:[a-z0-9]+)"#;
        // A word prefixed by a space.
        let _space_word = r#"(?: [a-z]+)"#;
        // A number prefixed by a space.
        let _space_number = r#"(?: [0-9]+)"#;
        // A number or a word prefixed by a space.
        let _space_word_or_number = r#"(?: [a-z0-9]+)"#;
        // Many words, numbers and capcode.
        let many_words_numbers_capcode = format!(
            "(?:(?:(?:{}| )?{})|(?:(?:{}| )?{})){{1,3}}",
            capcode, number, capcode, word
        );
        // Any lowercase word with capcode.
        let capcode_word = format!("(?:{}{})", capcode, word);
        // Any lowercase word with delete capcode.
        let delete_capcode_word = format!("(?:{}{})", delete_capcode, word);
        // Any number with capcode.
        // let capcode_number = format!("(?:{}{})", capcode, number);
        // Any number with delete capcode.
        let delete_capcode_number = format!("(?:{}{})", delete_capcode, number);
        // Any sequence of Chinese characters.
        let chinese = r#"(?:^[\u3400-\u4DBF\u4E00-\u9FFF]+)"#;

        let chinese_regex = Regex::new(&format!("^{}$", chinese)).unwrap();
        println!("Chinese: {}", chinese_regex);
        assert_regex_matches(
            &chinese_regex,
            &["好", "你好", "我叫罗杰斯"],
            &["hello", "你好 ", "好 "],
        );

        let capcode_regex = Regex::new(&format!("^{}$", capcode)).unwrap();
        println!("Capcode: {}", capcode_regex);
        assert_regex_matches(&capcode_regex, &["DC ", "DU ", "D ", "U ", "C "], &[" "]);

        let capcode_word_regex = Regex::new(&format!("^{}$", capcode_word)).unwrap();
        println!("Capcode lowercase word: {}", capcode_word_regex);
        assert_regex_matches(
            &capcode_word_regex,
            &["D github", "DU hi", "DC bonjour"],
            &["hello", "DC "],
        );

        let many_words_numbers_capcode_regex =
            Regex::new(&format!("^{}$", many_words_numbers_capcode)).unwrap();
        println!(
            "Many words, numbers and capcode: {}",
            many_words_numbers_capcode_regex
        );
        assert_regex_matches(
            &many_words_numbers_capcode_regex,
            &[
                "word",
                " word",
                "some words",
                " some more words",
                " some 123",
                "DC someU 123",
                "DC someU 123 word",
                "D someDC varDC 123",
            ],
            &["github ", "123 ", "DC"],
        );

        // ---------------------------------------------------------------------
        // URLs and Network Miscellaneous
        // ---------------------------------------------------------------------
        // A subdomain which is a sequence of {capcode}{word} which may be
        // separated by capcode.
        let subdomain = format!("(?:{}(?:-{})*)", delete_capcode_word, delete_capcode_word);
        // Multiple subdomains separated by dots.
        let subdomains = format!("(?:{}(?:\\.{})*)", subdomain, subdomain);
        // IPv4 address including the capcode.
        let ipv4 = format!(
            "(?:{}?{}(?:\\.{}){{3}})",
            delete_capcode, number, delete_capcode_number
        );
        // Port specification including the ':'.
        let port = format!("(?::{}[0-9]{{1,5}})", delete_capcode);

        let subdomain_regex = Regex::new(&format!("^{}$", subdomain)).unwrap();
        println!("Subdomain: {}", subdomain_regex);
        assert_regex_matches(
            &subdomain_regex,
            &["D github", "D github-D com", "D github-D com-D io"],
            &["D github-D com-D io-", "github-D com", "U github"],
        );

        let ipv4_regex = Regex::new(&format!("^{}$", ipv4)).unwrap();
        println!("IPv4: {}", ipv4_regex);
        assert_regex_matches(
            &ipv4_regex,
            &["D 127.D 0.D 0.D 1", "0.D 0.D 0.D 0"],
            &["D 256.D 256.D 256.D 256.D 256", "D 0.D 1."],
        );

        let subdomains_regex = Regex::new(&format!("^{}$", subdomains)).unwrap();
        println!("Subdomains: {}", subdomains_regex);
        assert_regex_matches(
            &subdomains_regex,
            &["D github-D world.D com", "D github-D world.D com-D io"],
            &["D github.D com."],
        );

        let port_regex = Regex::new(&format!("^ ?{}$", port)).unwrap();
        println!("Port: {}", port_regex);
        assert_regex_matches(
            &port_regex,
            &[":D 80", " :D 443", ":D 8080"],
            &[":D 999999"],
        );

        let space_and_url = format!("(?: {}://(?:{}|{}){}?)", word, subdomains, ipv4, port);

        let space_and_url_regex = Regex::new(&format!("^{}$", space_and_url)).unwrap();
        println!("Space and URL: {}", space_and_url_regex);
        assert_regex_matches(
            &space_and_url_regex,
            &[
                " https://D github.D com",
                " tcp://D www.D google.D com",
                " https://D grafana.D codegeex.D cn:D 8443",
                " http://D 127.D 0.D 0.D 1:D 80",
            ],
            &["https://github.com", " http://www"],
        );

        // ---------------------------------------------------------------------
        // HTML
        // ---------------------------------------------------------------------
        let html_tag = r#"(?:<D?[UC]? [a-z]+(?:>|/>| />)?)"#;

        let html_tag_regex = Regex::new(format!("^{}$", html_tag).as_str()).unwrap();
        println!("HTML Tag: {}", html_tag_regex);
        assert_regex_matches(
            &html_tag_regex,
            &["<D div>", "<DU a", "<D a/>", "<D a />"],
            &[],
        );

        // ---------------------------------------------------------------------
        // Punctuation
        // ---------------------------------------------------------------------
        // Punctuation with space.
        let many_punct_capcode_or_space = r#"(?:[[:punct:] DCU]+)"#;
        // L punctuation with whitespace.
        let l_punct_whitespace = format!("(?:{}?(?:[[:space:]]*))", many_punct_capcode_or_space);
        // R punctuation with whitespace.
        let r_punct_whitespace = format!("(?:[[:space:]]*{}?)", many_punct_capcode_or_space);
        // Word that begins with punctuation.
        let word_lpunct = format!(
            "(?:(?:[[:punct:]]|(?:::)){}{})",
            delete_capcode, word_or_number
        );

        let word_lpunct_regex = Regex::new(&format!("^{}$", word_lpunct)).unwrap();
        println!("Word with R punctuation: {}", word_lpunct_regex);
        assert_regex_matches(
            &word_lpunct_regex,
            &[
                "-D word", "_D 123", "*DU word", "/DC word", "&D word", "::D word",
            ],
            &["D word", "D word-", "D word_"],
        );

        let l_punct_whitespace_regex = Regex::new(&format!("^{}$", l_punct_whitespace)).unwrap();
        println!(
            "L Punctuation with whitespace: {}",
            l_punct_whitespace_regex
        );
        assert_regex_matches(
            &l_punct_whitespace_regex,
            &[" ", " . ", ".", " .", ". / ^ \t\n"],
            &[" \t \n . / ^ "],
        );

        let r_punct_whitespace_regex = Regex::new(&format!("^{}$", r_punct_whitespace)).unwrap();
        println!(
            "R Punctuation with whitespace: {}",
            r_punct_whitespace_regex
        );
        assert_regex_matches(
            &r_punct_whitespace_regex,
            &[" ", " . ", ".", " .", "\t\n./^"],
            &[". / ^ \t \n "],
        );

        // ---------------------------------------------------------------------
        // STRICT
        // ---------------------------------------------------------------------
        // The strict regex allows the following patterns:
        // - Lone characters
        // - Words
        // - Numbers
        // - Word with capcode
        // - Numbers with capcode
        // - Word or number LPunct
        // - R or L punct with whitespace
        // - Chinese
        // - URLs
        // - HTML tags
        let strict_regex = Regex::new(
            &[
                ".",
                &word_lpunct,
                &l_punct_whitespace,
                &r_punct_whitespace,
                &chinese,
                &space_and_url,
                &html_tag,
                &many_words_numbers_capcode,
            ]
            .map(|re| format!("(?:^{}$)", re))
            .join("|"),
        )
        .unwrap();
        println!("Strict: {}", strict_regex);
        assert_regex_matches(
            &strict_regex,
            &[
                // Capcode
                "DC ",
                // Simple word & number
                "word",
                "123",
                // Word & number with capcode
                "DC word",
                "DU 123",
                // Many words, numbers and capcode
                "DC someDU complexDU 123",
                // Word or number prefixed with punctuation
                "-D word",
                "_D 123",
                // Punctuation with whitespace
                "\n}",
                // Spaced URLs
                " https://D grafana.D codegeex.D cn:D 8443",
                // HTML
                "<D div",
                // Chinese
                "我叫罗杰斯",
            ],
            &[
                "github ",
                "123 ",
                "https://github.com",
                "hello/",
                "more than three words bad",
            ],
        );
    }

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
