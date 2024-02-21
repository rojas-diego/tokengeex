// Code imported and modified from: https://github.com/huggingface/tokenizers
// License: https://github.com/huggingface/tokenizers/blob/4a8105c36671ef46738d6e2799c55198139b87b2/LICENSE

use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet},
};

use derive_builder::Builder;
use rand::Rng;
use serde::{
    de::{MapAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Serialize, Serializer,
};

use crate::{
    capcode,
    core::Model,
    parallelism::{current_num_threads, MaybeParallelBridge, MaybeParallelSlice},
};

mod lattice;
mod trie;

use trie::Trie;

use self::{lattice::Lattice, trie::TrieBuilder};

type Error = Box<dyn std::error::Error + Send + Sync>;
type Result<T> = std::result::Result<T, Error>;

const BYTE_FALLBACKS: [&str; 256] = [
    "\0", "\x01", "\x02", "\x03", "\x04", "\x05", "\x06", "\x07", "\x08", "\x09", "\x0a", "\x0b",
    "\x0c", "\x0d", "\x0e", "\x0f", "\x10", "\x11", "\x12", "\x13", "\x14", "\x15", "\x16", "\x17",
    "\x18", "\x19", "\x1a", "\x1b", "\x1c", "\x1d", "\x1e", "\x1f", "\x20", "!", "\"", "#", "$",
    "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7",
    "8", "9", ":", ";", "<", "=", ">", "?", "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "\\", "]",
    "^", "_", "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
    "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "\x7f", "<0x80>",
    "<0x81>", "<0x82>", "<0x83>", "<0x84>", "<0x85>", "<0x86>", "<0x87>", "<0x88>", "<0x89>",
    "<0x8a>", "<0x8b>", "<0x8c>", "<0x8d>", "<0x8e>", "<0x8f>", "<0x90>", "<0x91>", "<0x92>",
    "<0x93>", "<0x94>", "<0x95>", "<0x96>", "<0x97>", "<0x98>", "<0x99>", "<0x9a>", "<0x9b>",
    "<0x9c>", "<0x9d>", "<0x9e>", "<0x9f>", "<0xa0>", "<0xa1>", "<0xa2>", "<0xa3>", "<0xa4>",
    "<0xa5>", "<0xa6>", "<0xa7>", "<0xa8>", "<0xa9>", "<0xaa>", "<0xab>", "<0xac>", "<0xad>",
    "<0xae>", "<0xaf>", "<0xb0>", "<0xb1>", "<0xb2>", "<0xb3>", "<0xb4>", "<0xb5>", "<0xb6>",
    "<0xb7>", "<0xb8>", "<0xb9>", "<0xba>", "<0xbb>", "<0xbc>", "<0xbd>", "<0xbe>", "<0xbf>",
    "<0xc0>", "<0xc1>", "<0xc2>", "<0xc3>", "<0xc4>", "<0xc5>", "<0xc6>", "<0xc7>", "<0xc8>",
    "<0xc9>", "<0xca>", "<0xcb>", "<0xcc>", "<0xcd>", "<0xce>", "<0xcf>", "<0xd0>", "<0xd1>",
    "<0xd2>", "<0xd3>", "<0xd4>", "<0xd5>", "<0xd6>", "<0xd7>", "<0xd8>", "<0xd9>", "<0xda>",
    "<0xdb>", "<0xdc>", "<0xdd>", "<0xde>", "<0xdf>", "<0xe0>", "<0xe1>", "<0xe2>", "<0xe3>",
    "<0xe4>", "<0xe5>", "<0xe6>", "<0xe7>", "<0xe8>", "<0xe9>", "<0xea>", "<0xeb>", "<0xec>",
    "<0xed>", "<0xee>", "<0xef>", "<0xf0>", "<0xf1>", "<0xf2>", "<0xf3>", "<0xf4>", "<0xf5>",
    "<0xf6>", "<0xf7>", "<0xf8>", "<0xf9>", "<0xfa>", "<0xfb>", "<0xfc>", "<0xfd>", "<0xfe>",
    "<0xff>",
];

pub type ScoredToken = (String, f64);

#[derive(thiserror::Error, Debug)]
pub enum UnigramError {
    #[error("vocabulary too small")]
    VocabularyTooSmall,
    #[error("missing token for byte value {0}")]
    MissingByteToken(u8),
}

#[derive(Clone)]
pub struct Unigram {
    vocab: Vec<ScoredToken>,
    token_to_ids: HashMap<String, u32>,
    trie: Trie<u8>,
}

impl Unigram {
    /// Create a new `Unigram` model from a vocabulary. The vocabulary is
    /// expected to contain at least 256 entries which correspond to the byte
    /// fallback entries.
    pub fn from(vocab: Vec<(String, f64)>) -> Result<Self> {
        let mut token_to_ids: HashMap<String, u32> = HashMap::new();
        let mut trie_builder = TrieBuilder::default();

        // Ensure the first 256 entries are byte fallbacks.
        if vocab.len() < 256 {
            return Err(UnigramError::VocabularyTooSmall.into());
        }
        for (id, (token, _)) in vocab.iter().enumerate().take(256) {
            if token != BYTE_FALLBACKS[id] {
                return Err(UnigramError::MissingByteToken(id as u8).into());
            }

            token_to_ids.insert(token.clone(), id as u32);

            if id < 128 {
                trie_builder.push(&[id as u8]);
            }
        }

        for (id, (token, _)) in vocab.iter().enumerate().skip(256) {
            token_to_ids.insert(token.into(), id as u32);

            let bytes: Vec<u8> = token.bytes().collect();
            trie_builder.push(&bytes);
        }

        let trie = trie_builder.build();

        Ok(Self {
            vocab,
            token_to_ids,
            trie,
        })
    }

    /// Populates a lattice with all the possible tokenizations of the input
    /// sentence.
    pub(super) fn populate_nodes(&self, lattice: &mut Lattice) {
        for (pos, c) in lattice.sentence.char_indices() {
            let suffix = lattice.sentence.bytes().skip(pos);

            // Search the available tokens for the current offset.
            let mut matches = self.trie.common_prefix_search(suffix).peekable();

            // There's no entry in the vocab that starts with suffix. We
            // fallback to byte level tokenization.
            if matches.peek().is_none() {
                let mut dst = [0; 4];
                let char_bytes = c.encode_utf8(&mut dst).as_bytes().iter().enumerate();

                for (i, &byte) in char_bytes {
                    let score = &self.vocab[byte as usize].1;

                    lattice.insert(pos + i, 1, *score, byte as usize);
                }
            } else {
                for token in matches {
                    let token = String::from_utf8(token).unwrap();
                    let token_id = *self.token_to_ids.get(&token).unwrap_or_else(|| {
                        panic!(
                            "expected token {:?} to exist in the vocab: {:?}",
                            token, self.vocab
                        )
                    });
                    let score = &self.vocab[token_id as usize].1;

                    lattice.insert(pos, token.len(), *score, token_id as usize);
                }
            }
        }
    }

    /// Iterate of vocabulary of the model as a pair of `(token, score)`.
    pub(super) fn iter(&self) -> UnigramIterator {
        UnigramIterator { model: self, i: 0 }
    }
}

impl Default for Unigram {
    /// Creates a default Unigram model with an empty vocabulary.
    fn default() -> Self {
        let mut vocab = vec![];

        for &fallback in BYTE_FALLBACKS.iter() {
            vocab.push((fallback.to_string(), 1.0 / 256.0));
        }

        Self::from(vocab).unwrap()
    }
}

/// Iterator to iterate of vocabulary of the model, and their relative score.
pub struct UnigramIterator<'a> {
    model: &'a Unigram,
    i: usize,
}

impl<'a> Iterator for UnigramIterator<'a> {
    type Item = &'a (String, f64);

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.i;
        if i < self.model.vocab_size() {
            let r = Some(&self.model.vocab[i]);
            self.i += 1;
            r
        } else {
            None
        }
    }
}

impl Model for Unigram {
    fn encode(&self, input: &str) -> Vec<u32> {
        #[derive(Clone, Debug)]
        struct Node {
            id: u32,
            score: f64,
            start: Option<usize>,
        }

        // For each position i in dp, we store the best tokenization of the
        // input sequence up to position i.
        let mut dp = vec![
            Node {
                id: 0,
                score: 0.0,
                start: None
            };
            input.len() + 1
        ];

        dp[0].start = Some(0);

        for (pos, c) in input.char_indices() {
            // We skip positions that are unreachable.
            if dp[pos].start.is_none() {
                continue;
            }

            let suffix = input.bytes().skip(pos);

            log::trace!(
                "Encoding pos={} suffix={:?} dp={:?}",
                pos,
                &input[pos..],
                dp
            );

            // Search the available tokens for the current offsetition.
            let mut matches = self.trie.common_prefix_search(suffix).peekable();

            // There's no entry in the vocab that starts with suffix. We
            // fallback to byte level tokenization.
            if matches.peek().is_none() {
                let mut char_bytes = [0; 4];
                let char_bytes = c.encode_utf8(&mut char_bytes).as_bytes().iter().enumerate();

                log::trace!("No matches for {:?}", &input[pos..]);

                for (i, &byte) in char_bytes {
                    log::trace!("Byte: {}", byte);

                    // The node at which this byte ends.
                    let pos = pos + i;
                    let score = dp[pos].score;
                    let dst_node = &dp[pos + 1];
                    let token_score = self.vocab[byte as usize].1;
                    let new_score = score + token_score;

                    if dst_node.start.is_none() || new_score > dst_node.score {
                        dp[pos + 1] = Node {
                            id: byte as u32,
                            score: new_score,
                            start: Some(pos),
                        };
                    }
                }
            // Otherwise, we Gucci and we can just iterate over the matches.
            } else {
                for token in matches {
                    log::trace!("Match token={:?}", String::from_utf8_lossy(&token));

                    // The node at which this token ends.
                    let dst_node = &dp[pos + token.len()];
                    let token = String::from_utf8(token).unwrap();
                    let token_id = *self.token_to_ids.get(&token).unwrap();
                    let token_score = self.vocab[token_id as usize].1;
                    let new_score = dp[pos].score + token_score;

                    if dst_node.start.is_none() || new_score > dst_node.score {
                        log::trace!(
                            "New best score score={} range={}..{}",
                            new_score,
                            pos,
                            pos + token.len()
                        );

                        dp[pos + token.len()] = Node {
                            id: token_id,
                            score: new_score,
                            start: Some(pos),
                        };
                    }
                }
            }
        }

        // Backtrack along the best path to recover the tokens.
        let mut pos = input.len();
        let mut ids: Vec<u32> = vec![];

        log::trace!("Encode reached end dp={:?}", dp);

        while pos > 0 {
            let node = &dp[pos];

            let start = node.start.unwrap_or_else(|| {
                panic!(
                    "decode: current node at pos {}/{} (id={}, score={}) has no start position",
                    pos,
                    input.len(),
                    node.id,
                    node.score,
                )
            });

            ids.push(node.id);
            pos = start;
        }

        // Reverse to maintain original order since we built it backwards.
        ids.reverse();

        ids
    }

    fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|id| self.vocab[*id as usize].0.as_str())
            .collect::<Vec<&str>>()
            .join("")
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_ids.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab.get(id as usize).map(|item| item.0.clone())
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

impl Serialize for Unigram {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut model = serializer.serialize_struct("Unigram", 1)?;

        model.serialize_field("type", "unigram")?;
        model.serialize_field("vocab", &self.vocab)?;

        model.end()
    }
}

impl<'de> Deserialize<'de> for Unigram {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
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

    fn visit_map<V>(self, mut map: V) -> std::result::Result<Self::Value, V::Error>
    where
        V: MapAccess<'de>,
    {
        let mut model_type: Option<String> = None;
        let mut vocab: Option<Vec<ScoredToken>> = None;

        while let Some(key) = map.next_key()? {
            match key {
                "type" => {
                    model_type = Some(map.next_value()?);
                }
                "vocab" => {
                    vocab = Some(map.next_value()?);
                }
                _ => {
                    return Err(serde::de::Error::unknown_field(key, &["type"]));
                }
            }
        }

        match model_type.as_deref() {
            Some("unigram") => match vocab {
                Some(vocab) => Ok(Unigram::from(vocab).unwrap()),
                None => Ok(Unigram::default()),
            },
            _ => Err(serde::de::Error::custom("invalid model type")),
        }
    }
}

/// A sentence represents a string and its frequency in the training dataset.
pub type Sentence = (String, u32);

#[derive(Debug, Clone)]
/// Sentence generator is used to generate sentences from the training dataset.
/// This might become a trait later but right now it just splits each sample in
/// the dataset in sentences of ~N characters.
pub struct SentenceGenerator {
    /// Max size of chunks in characters.
    max_sentence_size: usize,
}

impl SentenceGenerator {
    pub fn new(max_sentence_size: usize) -> Self {
        Self { max_sentence_size }
    }

    pub fn generate_sentences(&self, sample: &str) -> Vec<Sentence> {
        let mut sentences = Vec::new();
        let mut last_pos = 0;
        let mut last_whitespace_pos = 0;

        for (pos, c) in sample.char_indices() {
            if c.is_whitespace() {
                last_whitespace_pos = pos;
            }

            if pos - last_pos >= self.max_sentence_size {
                if last_whitespace_pos > last_pos {
                    sentences.push((sample[last_pos..last_whitespace_pos].to_string(), 1));
                    last_pos = last_whitespace_pos;
                } else {
                    sentences.push((sample[last_pos..pos].to_string(), 1));
                    last_pos = pos;
                }
            }
        }

        if last_pos < sample.len() {
            sentences.push((sample[last_pos..].to_string(), 1));
        }

        sentences
    }
}

#[derive(Debug, Clone)]
/// Vocabulary generator is used to construct the initial vocabulary from the
/// training dataset. This might become a trait later but right now it picks
/// the most common substrings in the dataset and scores them based on their
/// frequency and length. It includes the 256 byte fallbacks.
pub struct VocabularyGenerator {
    words_per_token: usize,
    window_size: usize,
    insert_probability: f64,
    strict: bool,
}

impl VocabularyGenerator {
    // TODO: Need to validate that insert probability is between 0 and 1.
    pub fn new(
        words_per_token: usize,
        window_size: usize,
        insert_probability: f64,
        strict: bool,
    ) -> Self {
        Self {
            words_per_token,
            window_size,
            insert_probability,
            strict,
        }
    }

    /// Generates an initial vocabulary from the given samples.
    pub fn generate_vocabulary(self, samples: &[&str], size: usize) -> Vec<ScoredToken> {
        let mut tokens = HashMap::new();
        let mut byte_freq = [0usize; 256];
        let mut rng = rand::thread_rng();

        for (i, sample) in samples.iter().enumerate() {
            log::info!(
                "{}/{}: Generating vocabulary from sample ({} tokens)",
                i + 1,
                samples.len(),
                tokens.len()
            );

            let mut sample_tokens = HashSet::new();

            for (i, _) in sample.char_indices() {
                let suffix = &sample[i..];
                for (ii, c) in suffix.char_indices().take(self.window_size) {
                    let candidate = &suffix[..ii + c.len_utf8()];

                    if self.is_valid_token(candidate)
                        && rng.gen_range(0.0..1.0) < self.insert_probability
                    {
                        sample_tokens.insert(candidate.to_string());
                    }
                }
            }

            for b in sample.bytes() {
                byte_freq[b as usize] += 1;
            }

            for token in sample_tokens {
                *tokens.entry(token).or_insert(0) += 1;
            }
        }

        log::info!("Sorting tokens by frequency");

        // Convert the tokens to a vector and sort them by frequency.
        let mut tokens: Vec<_> = tokens.into_iter().collect();
        tokens.sort_by_key(|(_, freq)| Reverse(*freq));

        // Keep track of duplicates, ensuring the earlier occurrence is kept.
        let mut seen = HashSet::new();

        // Add all 256 ASCII characters and byte values to the initial
        // vocabulary.
        let mut vocab: Vec<ScoredToken> = BYTE_FALLBACKS
            .iter()
            .enumerate()
            .map(|(i, token)| {
                seen.insert(token.to_string());

                (token.to_string(), byte_freq[i] as f64)
            })
            .collect();

        // We further add the most frequent substrings.
        for (token, freq) in tokens.into_iter() {
            if vocab.len() >= size {
                break;
            }

            if !seen.contains(&token) {
                vocab.push((token.to_string(), (freq * token.len()) as f64));
            }

            seen.insert(token);
        }

        // Convert the scores to log probabilities.
        to_log_prob(&mut vocab);

        // Sort the vocabulary by score.
        vocab[256..]
            .sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        vocab
    }

    /// Checks whether the given token is valid.
    pub fn is_valid_token(&self, token: &str) -> bool {
        if token.is_empty() {
            return false;
        }

        let mut has_word = false; // Any alphanumeric sequence.
        let mut has_ascii = false;
        let mut has_punctuation = false; // Anything non-alphanumeric.
        let mut has_non_ascii = false;
        let mut num_spaces = 0;
        let mut num_words = 0; // Number of whitespace separated sequences of characters.
        let mut previous_char = ' ';

        for c in token.chars() {
            if !capcode::is_marker(c) {
                if c.is_ascii() {
                    has_ascii = true;

                    if c.is_alphanumeric() {
                        has_word = true;
                    } else if c.is_whitespace() {
                        num_spaces += 1;
                    } else {
                        has_punctuation = true;
                    }
                } else {
                    has_non_ascii = true;
                }
            }

            if c.is_whitespace() && !previous_char.is_whitespace() {
                num_words += 1;
            }

            previous_char = c;
        }

        if num_words > self.words_per_token {
            return false;
        }

        if self.strict {
            if has_ascii && has_non_ascii {
                return false;
            }
            if has_word && (previous_char.is_whitespace() || num_spaces > 1 || has_punctuation) {
                return false;
            }
        }

        true
    }
}

#[non_exhaustive]
#[derive(Builder, Clone)]
pub struct UnigramTrainer {
    #[builder(default = "8000")]
    pub vocab_size: usize,
    #[builder(default = "2")]
    pub num_sub_iterations: usize,
    #[builder(default = "0.75")]
    pub shrinking_factor: f64,
    #[builder(default = "16")]
    pub max_token_length: usize,
    #[builder(default = "1_000_000")]
    pub initial_vocab_size: usize,
    #[builder(default = "HashSet::new()")]
    pub initial_alphabet: HashSet<char>,
    #[builder(default = "vec![]")]
    pub suggested_tokens: Vec<String>,
    #[builder(default = "vec![]")]
    pub added_tokens: Vec<String>,

    #[builder(default = "Vec::new()")]
    sentences: Vec<Sentence>,
}

impl Default for UnigramTrainer {
    fn default() -> Self {
        Self::builder().build().unwrap()
    }
}

impl UnigramTrainer {
    pub fn builder() -> UnigramTrainerBuilder {
        UnigramTrainerBuilder::default()
    }

    pub fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<Sentence>> + Sync,
    {
        let all_sentences: Result<HashMap<String, u32>> = iterator
            .maybe_par_bridge()
            .map(|sample| {
                let mut map = HashMap::new();
                let sentences = process(sample.as_ref())?;
                for (sentence, freq) in sentences {
                    map.entry(sentence)
                        .and_modify(|c| *c += freq)
                        .or_insert(freq);
                }
                Ok(map)
            })
            .reduce(
                || Ok(HashMap::new()),
                |acc, ws| {
                    let mut acc = acc?;
                    for (k, v) in ws? {
                        acc.entry(k).and_modify(|c| *c += v).or_insert(v);
                    }
                    Ok(acc)
                },
            );

        self.sentences.extend(all_sentences?);

        Ok(())
    }

    /// Train a Unigram model over a dataset of sentences using the initial
    /// vocabulary.
    pub fn train(&self, model: &mut Unigram, vocab: Vec<ScoredToken>) -> Result<()> {
        let mut vocab = vocab;

        log::info!(
            "Using {} pieces on {} sentences for EM training",
            vocab.len(),
            self.sentences.len()
        );

        let desired_vocab_size: usize = (self.vocab_size * 11) / 10; // * 1.1

        let expected_loops = (((desired_vocab_size as f64).ln() - (vocab.len() as f64).ln())
            / self.shrinking_factor.ln()) as usize
            + 1;

        log::info!(
            "Running {} EM loop(s) to fine grain the pieces",
            expected_loops
        );

        let mut new_model = Unigram::from(vocab.clone())?;

        let mut em_iter = 0;

        loop {
            for em_subiter in 0..self.num_sub_iterations {
                let (objective, num_tokens, num_bytes, expected) = self.run_e_step(&new_model);

                vocab = self.run_m_step(&vocab, &expected);

                log::info!(
                    "EM iter={} subiter={} previous_vocab_size={} vocab_size={} objective={} num_tokens={} compression={}",
                    em_iter,
                    em_subiter,
                    new_model.vocab_size(),
                    vocab.len(),
                    objective,
                    num_tokens,
                    (num_bytes as f64) / (num_tokens as f64)
                );

                new_model = Unigram::from(vocab.clone())?;
            }

            // Stops the iteration when the size of sentences reaches to the
            // desired symbol size.
            if vocab.len() <= desired_vocab_size {
                break;
            }

            // Prunes pieces.
            vocab = self.prune_vocab(&new_model, &vocab);
            new_model = Unigram::from(vocab.clone())?;
            em_iter += 1;
        }

        // Finally, adjusts the size of sentencepices to be |vocab_size|.
        *model = self.finalize(new_model)?;

        Ok(())
    }

    fn prune_vocab(&self, model: &Unigram, vocab: &[ScoredToken]) -> Vec<ScoredToken> {
        let mut always_keep = vec![true; vocab.len()];
        let mut alternatives: Vec<Vec<usize>> = vec![Vec::new(); vocab.len()];

        let bos_id = vocab.len() + 1;
        let eos_id = vocab.len() + 2;

        // First, segments the current sentencepieces to know
        // how each sentencepiece is resegmented if this sentencepiece is removed
        // from the vocabulary.
        // To do so, we take the second best segmentation of sentencepiece[i].
        // alternatives[i] stores the sequence of second best sentencepieces.
        for (id, (token, _)) in vocab.iter().enumerate() {
            // Always keep unk and byte fallback tokens.
            if id <= 256 {
                continue;
            }

            let mut lattice = Lattice::from(token, bos_id, eos_id);
            model.populate_nodes(&mut lattice);

            let nbests = lattice.nbest(2);
            if nbests.len() == 1 {
                always_keep[id] = true;
            } else if nbests[0].len() >= 2 {
                always_keep[id] = false;
            } else if nbests[0].len() == 1 {
                always_keep[id] = true;
                for node in &nbests[1] {
                    let alt_id = node.borrow().id;
                    alternatives[id].push(alt_id);
                }
            }
        }

        // Second, segments all sentences to compute likelihood
        // with a unigram language model. inverted[i] stores
        // the set of sentence indices where the sentencepieces[i] appears.
        let chunk_size = std::cmp::max(self.sentences.len() / current_num_threads(), 1);
        // FIX: Cloning the entire dataset is not necessary here.
        let indexed_sentences: Vec<(usize, &Sentence)> =
            self.sentences.iter().enumerate().collect();
        let collected: (f64, Vec<f64>, Vec<Vec<usize>>) = indexed_sentences
            .maybe_par_chunks(chunk_size)
            .map(|enumerated_sentence_count_chunk| {
                let mut vsum = 0.0;
                let mut freq: Vec<f64> = vec![0.0; vocab.len()];
                let mut inverted: Vec<Vec<usize>> = vec![Vec::new(); vocab.len()];

                for (i, (sentence, count)) in enumerated_sentence_count_chunk {
                    let mut lattice = Lattice::from(sentence, bos_id, eos_id);
                    model.populate_nodes(&mut lattice);
                    vsum += *count as f64;
                    for node_ref in lattice.viterbi() {
                        let id = node_ref.borrow().id;
                        freq[id] += *count as f64;
                        inverted[id].push(*i);
                    }
                }
                (vsum, freq, inverted)
            })
            .reduce(
                || (0.0, vec![0.0; vocab.len()], vec![Vec::new(); vocab.len()]),
                |(vsum, freq, inverted), (lvsum, lfreq, linverted)| {
                    (
                        vsum + lvsum,
                        freq.iter()
                            .zip(lfreq)
                            .map(|(global_el, local_el)| global_el + local_el)
                            .collect(),
                        inverted
                            .iter()
                            .zip(linverted)
                            .map(|(global_el, local_el)| [&global_el[..], &local_el[..]].concat())
                            .collect(),
                    )
                },
            );

        let (vsum, freq, inverted) = collected;

        let sum: f64 = freq.iter().sum();
        let logsum = sum.ln();
        let mut candidates: Vec<(usize, f64)> = vec![];
        let mut pruned_vocab: Vec<ScoredToken> = Vec::with_capacity(self.vocab_size);

        // Add the byte fallbacks and unknown token.
        for word in vocab.iter().take(256 + 1) {
            pruned_vocab.push(word.clone());
        }

        // Finally, computes how likely the LM likelihood is reduced if
        // the sentencepiece[i] is removed from the vocabulary.
        // Since the exact computation of loss is difficult, we compute the
        // loss approximately by assuming that all sentencepiece[i] in the sentences
        // are replaced with alternatives[i] when sentencepiece[i] is removed.
        for (id, (token, score)) in vocab.iter().enumerate() {
            if id <= 256 {
                continue;
            }

            if freq[id] == 0.0 && !always_keep[id] {
                // not found in Viterbi path. Can remove this entry safely.
                continue;
            } else if alternatives[id].is_empty() {
                // no alternatives. Keeps this entry.
                pruned_vocab.push((token.to_string(), *score));
            } else {
                let mut f = 0.0; // the frequency of pieces[i];

                for n in &inverted[id] {
                    let score = self.sentences[*n].1 as f64;
                    f += score;
                }

                if f == 0.0 || f.is_nan() {
                    continue;
                }

                f /= vsum; // normalizes by all sentence frequency.

                let logprob_sp = freq[id].ln() - logsum;

                // After removing the sentencepiece[i], its frequency freq[i] is
                // re-assigned to alternatives.
                // new_sum = current_sum - freq[i] + freq[i] * alternatives.size()
                //         = current_sum + freq[i] (alternatives - 1)
                let logsum_alt = (sum + freq[id] * (alternatives.len() - 1) as f64).ln();

                // The frequencies of altenatives are increased by freq[i].
                let mut logprob_alt = 0.0;
                for n in &alternatives[id] {
                    logprob_alt += (freq[*n] + freq[id]).ln() - logsum_alt;
                }

                // loss: the diff of likelihood after removing the sentencepieces[i].
                let loss = f * (logprob_sp - logprob_alt);
                assert!(!loss.is_nan());

                candidates.push((id, loss));
            }
        }

        let desired_vocab_size: usize = (self.vocab_size * 11) / 10; // * 1.1
        let pruned_size: usize = ((vocab.len() as f64) * self.shrinking_factor) as usize;
        let pruned_size = desired_vocab_size.max(pruned_size);

        candidates.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        for (id, _score) in candidates {
            if pruned_vocab.len() == pruned_size {
                break;
            }
            pruned_vocab.push(vocab[id].clone());
        }

        pruned_vocab.to_vec()
    }

    fn run_e_step(&self, model: &Unigram) -> (f64, u32, u32, Vec<f64>) {
        let sentence_frequencies_sum: u32 = self.sentences.iter().map(|(_a, b)| *b).sum();
        let chunk_size = std::cmp::max(self.sentences.len() / current_num_threads(), 1);

        let scores: (f64, u32, u32, Vec<f64>) = self.sentences
            .maybe_par_chunks(chunk_size)
            .map(|chunk| {
                let mut objs: f64 = 0.0;
                let mut expected: Vec<f64> = vec![0.0; model.vocab_size()];
                let mut num_tokens: u32 = 0;
                let mut num_bytes: u32 = 0;

                for (sentence, freq) in chunk {
                    let mut lattice =
                        Lattice::from(sentence, model.vocab.len() + 1, model.vocab.len() + 2);

                    model.populate_nodes(&mut lattice);

                    let z = lattice.populate_marginal(*freq as f64, &mut expected);

                    if z.is_nan() {
                        // Collect all the tokens that were in the lattice
                        let tokens = lattice
                            .begin_nodes
                            .iter()
                            .take(lattice.begin_nodes.len() - 1)
                            .flat_map(|node| node.iter().map(|n| n.borrow().id))
                            .collect::<HashSet<usize>>()
                            .iter()
                            .map(|id| (*id, model.vocab[*id].clone()))
                            .collect::<Vec<_>>();

                        panic!("marginal probaility for sentence {:?} is f64::NaN lattice={} tokens={:?}", sentence, lattice, tokens);
                    }

                    num_tokens += lattice.viterbi().len() as u32;
                    num_bytes += sentence.len() as u32;
                    objs -= z / (sentence_frequencies_sum as f64);
                }
                (objs, num_tokens, num_bytes, expected)
            })
            .reduce(
                || (0.0, 0, 0, vec![0.0; model.vocab_size()]),
                |(objs, ntokens, nbytes, expected), (lobjs, lntokens, lnbytes, lexpected)| {
                    (
                        objs + lobjs,
                        ntokens + lntokens,
                        nbytes + lnbytes,
                        expected
                            .iter()
                            .zip(lexpected)
                            .map(|(global_el, local_el)| global_el + local_el)
                            .collect(),
                    )
                },
            );

        scores
    }

    fn run_m_step(&self, pieces: &[ScoredToken], expected: &[f64]) -> Vec<ScoredToken> {
        assert_eq!(pieces.len(), expected.len());

        let mut alternative_vocab: Vec<ScoredToken> = Vec::with_capacity(self.vocab_size);

        let mut sum = 0.0;
        let expected_frequency_threshold = 0.5;

        for (i, (freq, (piece, _score))) in expected.iter().zip(pieces).enumerate() {
            // We always keep the first 256 entries which correspond to our byte
            // fallbacks.
            if *freq < expected_frequency_threshold && i > 256 {
                continue;
            }

            alternative_vocab.push((piece.clone(), *freq));
            sum += freq;
        }

        // I have no clue what's going here. A previous comment pointed to this
        // paper:
        // https://cs.stanford.edu/~pliang/papers/tutorial-acl2007-talk.pdf
        let logsum = digamma(sum);
        let alternative_vocab: Vec<_> = alternative_vocab
            .into_iter()
            .map(|(s, c)| (s, digamma(c) - logsum))
            .collect();

        alternative_vocab
    }

    fn finalize(&self, model: Unigram) -> Result<Unigram> {
        let mut vocab: Vec<(String, f64)> = vec![];

        let vocab_size_without_special_tokens = self.vocab_size;

        for (token, score) in model.iter() {
            vocab.push((
                token.to_string(),
                if !score.is_normal() { 0.0 } else { *score },
            ));

            if vocab.len() == vocab_size_without_special_tokens {
                break;
            }
        }

        vocab[256..].sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        Unigram::from(vocab.into_iter().collect())
    }
}

fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 7.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    x -= 1.0 / 2.0;
    let xx = 1.0 / x;
    let xx2 = xx * xx;
    let xx4 = xx2 * xx2;
    result += x.ln() + (1.0 / 24.0) * xx2 - 7.0 / 960.0 * xx4 + (31.0 / 8064.0) * xx4 * xx2
        - (127.0 / 30720.0) * xx4 * xx4;
    result
}

fn to_log_prob(pieces: &mut [ScoredToken]) {
    let sum: f64 = pieces.iter().map(|(_, score)| score).sum();
    let logsum = sum.ln();
    for (_, score) in pieces.iter_mut() {
        *score = score.ln() - logsum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_to_log_prob() {
        let mut a = vec![("".to_string(), 1.0), ("".to_string(), 2.0)];
        to_log_prob(&mut a);
        let scores = a.iter().map(|(_, score)| *score).collect::<Vec<_>>();
        // ln(1) - ln(3)
        assert_approx_eq!(scores[0], -1.098, 0.01);
        // ln(2) - ln(3)
        assert_approx_eq!(scores[1], -0.405, 0.01);

        let mut a = vec![("".to_string(), 0.0), ("".to_string(), 3.0)];
        to_log_prob(&mut a);
        let scores = a.iter().map(|(_, score)| *score).collect::<Vec<_>>();
        // ln(0) - ln(3)
        assert_eq!(scores[0], f64::NEG_INFINITY);
    }

    #[test]
    fn test_vocab_generator_is_valid_token() {
        // Strict
        let vg = VocabularyGenerator::new(2, 2, 0.0, true);

        assert!(vg.is_valid_token("hello"));
        assert!(!vg.is_valid_token("hello "));
        assert!(vg.is_valid_token("hello world"));
        assert!(!vg.is_valid_token("hello world "));

        assert!(vg.is_valid_token(" abc"));
        assert!(vg.is_valid_token(" 123"));

        assert!(vg.is_valid_token("大家哦好"));
        assert!(!vg.is_valid_token("Hello 大家哦好"));
        assert!(!vg.is_valid_token("été"));

        assert!(vg.is_valid_token("// ****"));
        assert!(vg.is_valid_token("//D"));
        assert!(vg.is_valid_token("D "));

        assert!(vg.is_valid_token(" + "));
        assert!(vg.is_valid_token(" +D "));

        assert!(vg.is_valid_token("    "));
        assert!(vg.is_valid_token("\n"));
        assert!(vg.is_valid_token("\t"));
        assert!(vg.is_valid_token("\n\t"));
        assert!(vg.is_valid_token("\n\t\t"));

        assert!(!vg.is_valid_token("<div>"));
        assert!(!vg.is_valid_token("(D self"));

        let vg = VocabularyGenerator::new(2, 2, 0.0, false);

        assert!(vg.is_valid_token("D "));
        assert!(vg.is_valid_token("<div>"));
        assert!(vg.is_valid_token("(D self"));
    }

    #[test]
    fn test_sentence_generator() {
        let generator = SentenceGenerator::new(12);
        let sentences = generator.generate_sentences("Hello world");
        assert_eq!(sentences, vec![("Hello world".into(), 1)]);

        let sentences = generator.generate_sentences("Hello world, how are you?");
        assert_eq!(
            sentences,
            vec![
                ("Hello world,".into(), 1),
                (" how are".into(), 1),
                (" you?".into(), 1)
            ]
        );

        let generator = SentenceGenerator::new(1);
        let sentences = generator.generate_sentences(" 1\r\n2 3");
        assert_eq!(
            sentences,
            vec![
                (" ".into(), 1),
                ("1".into(), 1),
                ("\r".into(), 1),
                ("\n".into(), 1),
                ("2".into(), 1),
                (" ".into(), 1),
                ("3".into(), 1)
            ]
        );
    }

    use serde_json;

    #[test]
    fn serialize_unigram() {
        let unigram = Unigram::default();
        let serialized = serde_json::to_string(&unigram).unwrap();
        assert_eq!(serialized, r#"{"type":"unigram"}"#);
    }

    #[test]
    fn deserialize_unigram() {
        let deserialized: std::result::Result<Unigram, serde_json::Error> =
            serde_json::from_str(r#"{"type":"unigram"}"#);
        assert!(deserialized.is_ok());

        let deserialized: std::result::Result<Unigram, serde_json::Error> =
            serde_json::from_str(r#"{"type":"bigram"}"#);
        assert!(deserialized.is_err());
    }
}
