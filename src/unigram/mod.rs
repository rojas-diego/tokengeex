// Code imported and modified from: https://github.com/huggingface/tokenizers
// License: https://github.com/huggingface/tokenizers/blob/4a8105c36671ef46738d6e2799c55198139b87b2/LICENSE

use crate::{
    utils::{lattice::Lattice, trie::Trie},
    Error, Model, ModelWrapper, Result, ScoredToken, Token, TokenID,
};
use std::collections::HashMap;

mod serialization;
mod trainer;

pub use serialization::Vocab;
pub use trainer::*;

#[derive(Copy, Clone)]
pub enum UnigramError {
    NoPath(usize, usize),
    TokenIdOutOfBounds(u32),
}

impl std::fmt::Display for UnigramError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            UnigramError::NoPath(pos, len) => {
                write!(f, "no path to position {}/{}", pos, len)
            }
            UnigramError::TokenIdOutOfBounds(id) => {
                write!(f, "token id {} is out of bounds", id)
            }
        }
    }
}

impl std::fmt::Debug for UnigramError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            UnigramError::NoPath(pos, len) => {
                write!(f, "NoPath({}, {})", pos, len)
            }
            UnigramError::TokenIdOutOfBounds(id) => {
                write!(f, "TokenIdOutOfBounds({})", id)
            }
        }
    }
}

impl std::error::Error for UnigramError {}

#[derive(Clone, Default)]
/// Unigram is a tokenization model that uses a vocabulary of scored tokens to
/// encode and decode sequences. It is based on the UnigramLM algorithm
/// described in https://arxiv.org/abs/1804.10959.
/// This implementation of UnigramLM is mostly based on the SentencePiece
/// implementation, however, it considers tokens to be byte sequences instead of
/// Unicode string. This is because we wish "byte fallback" mechanisms to not be
/// a special case but rather a natural part of the model.
pub struct Unigram {
    vocab: Vec<ScoredToken>,
    token_to_ids: HashMap<Token, u32>,
    trie: Trie<(TokenID, u32)>,
    pub capcode: bool,
}

impl From<Unigram> for ModelWrapper {
    fn from(val: Unigram) -> Self {
        ModelWrapper::Unigram(val)
    }
}

impl Unigram {
    /// Create a new `Unigram` model from a vocabulary of scored tokens.
    pub fn from(vocab: Vec<ScoredToken>, capcode: bool) -> Self {
        let mut token_to_ids: HashMap<Token, u32> = HashMap::new();
        let mut trie = Trie::default();

        for (id, (token, _)) in vocab.iter().enumerate() {
            token_to_ids.insert(token.clone(), id as u32);
            trie.push(token, (id as u32, token.len() as u32));
        }

        Self {
            vocab,
            token_to_ids,
            trie,
            capcode,
        }
    }

    /// Access the vocabulary of the model.
    pub fn vocab(&self) -> &Vec<ScoredToken> {
        &self.vocab
    }

    /// Populates a lattice with all the possible tokenizations of the input
    /// sentence.
    pub fn populate_nodes(&self, lattice: &mut Lattice) {
        let mut buff = Vec::<u8>::with_capacity(256);
        let input = lattice.sentence;

        let delete_marker_id = self.token_to_id("D");
        let delete_marker_score = if let Some(delete_marker_token_id) = delete_marker_id {
            self.vocab[delete_marker_token_id as usize].1
        } else {
            0.0
        };

        assert!(self.capcode == delete_marker_id.is_some());

        for pos in 0..input.len() {
            let suffix = &input[pos..];

            // "two|words" -> "two", "words"
            // "two|words" -> "two", "D words"
            // "two|words" -> "two", "D", " words"
            for (prefix, delete, penalty) in &[
                (b"".as_slice(), false, 0.0),
                (b"D ".as_slice(), false, 0.0),
                (b" ".as_slice(), true, delete_marker_score),
            ] {
                if !prefix.is_empty() {
                    if !self.capcode || pos == 0 {
                        break;
                    }

                    let prev_char = char::from_u32(input[pos - 1] as u32);
                    let curr_char = char::from_u32(input[pos] as u32);

                    for char in [prev_char, curr_char] {
                        match char {
                            Some(c) => {
                                if !c.is_lowercase() && !c.is_ascii_digit() {
                                    break;
                                }
                            }
                            None => {
                                break;
                            }
                        }
                    }
                }

                buff.clear();
                for (id, len) in self
                    .trie
                    .common_prefix_search(prefix.iter().chain(suffix.iter()).copied(), &mut buff)
                    .filter(|(_, len)| *len > prefix.len() as u32)
                    .map(|(id, len)| (id, (len as usize) - prefix.len()))
                {
                    let score = &self.vocab[id as usize].1;

                    lattice.insert(pos, id, len, *score + penalty, *delete);
                }
            }
        }
    }
}

impl Model for Unigram {
    /// Encode the input sequence into a sequence of token IDs in O(n) time
    /// using the SentencePiece DP algorithm.
    fn encode(&self, input: &str) -> Result<Vec<u32>> {
        let mut buff = Vec::<u8>::with_capacity(256);
        let input = input.as_bytes();

        let delete_marker_id = self.token_to_id("D");
        let delete_marker_score = if let Some(delete_marker_token_id) = delete_marker_id {
            self.vocab[delete_marker_token_id as usize].1
        } else {
            0.0
        };

        #[derive(Clone, Debug)]
        struct Node {
            id: u32,
            score: f64,
            start: Option<usize>,
            delete: bool,
        }

        // For each position i in dp, we store the best tokenization of the
        // input sequence up to position i.
        let mut dp = vec![
            Node {
                id: 0,
                score: 0.0,
                start: None,
                delete: false,
            };
            input.len() + 1
        ];

        dp[0].start = Some(0);

        for pos in 0..input.len() {
            // We skip positions that are unreachable.
            if dp[pos].start.is_none() {
                continue;
            }

            let suffix = &input[pos..];

            // "two|words" -> "two", "words"
            // "two|words" -> "two", "D words"
            // "two|words" -> "two", "D", " words"
            for (prefix, delete, penalty) in &[
                (b"".as_slice(), false, 0.0),
                (b"D ".as_slice(), false, 0.0),
                (b" ".as_slice(), true, delete_marker_score),
            ] {
                if !prefix.is_empty() {
                    if !self.capcode || pos == 0 {
                        break;
                    }

                    let prev_char = char::from_u32(input[pos - 1] as u32);
                    let curr_char = char::from_u32(input[pos] as u32);

                    for char in [prev_char, curr_char] {
                        match char {
                            Some(c) => {
                                if !c.is_lowercase() && !c.is_ascii_digit() {
                                    break;
                                }
                            }
                            None => {
                                break;
                            }
                        }
                    }
                }

                buff.clear();
                for (id, len) in self
                    .trie
                    .common_prefix_search(prefix.iter().chain(suffix.iter()).copied(), &mut buff)
                    .filter(|(_, len)| *len > prefix.len() as u32)
                    .map(|(id, len)| (id, (len as usize) - prefix.len()))
                {
                    let node = &dp[pos + len];
                    let score = dp[pos].score + self.vocab[id as usize].1;

                    if node.start.is_none() || score > node.score {
                        dp[pos + len] = Node {
                            id,
                            score: score + penalty,
                            start: Some(pos),
                            delete: *delete,
                        };
                    }
                }
            }
        }

        // Backtrack along the best path to recover the tokens.
        let mut pos = input.len();
        let mut ids: Vec<u32> = Vec::with_capacity(input.len() / 2);

        while pos > 0 {
            let node = &dp[pos];

            let start = node
                .start
                .ok_or_else(|| Box::new(UnigramError::NoPath(pos, input.len())) as Error)?;

            ids.push(node.id);

            if node.delete {
                ids.push(
                    delete_marker_id.expect("delete marker 'D' to exist when capcode is enabled"),
                );
            }

            pos = start;
        }

        // Reverse to maintain original order since we built it backwards.
        ids.reverse();

        Ok(ids)
    }

    /// Decode the input sequence of token IDs into a string in O(n) time. If
    /// the string is not valid UTF-8, it will be returned as a lossy string.
    ///
    /// # Panics
    ///
    /// This method will panic if any of the token IDs are out of bounds.
    fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut res = Vec::new();

        for &id in ids {
            if id >= self.vocab_size() as u32 {
                return Err(Box::new(UnigramError::TokenIdOutOfBounds(id)));
            }

            let (token, _) = &self.vocab[id as usize];

            res.extend_from_slice(token);
        }

        Ok(String::from_utf8_lossy(&res).into_owned())
    }

    /// Convert a token to a token ID. Currently it is not possible to access
    /// tokens that are invalid UTF-8 through this method.
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_ids.get(token.as_bytes()).copied()
    }

    /// Convert a token ID to a token. If the byte sequence is not valid UTF-8
    /// it will be returned as a lossy string.
    fn id_to_token(&self, id: u32) -> Option<String> {
        if id > self.vocab.len() as u32 {
            return None;
        }

        Some(String::from_utf8_lossy(&self.vocab[id as usize].0).into_owned())
    }

    /// Number of entries in the vocabulary.
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode() {
        let vocab = [("a", 1.0), ("b", 2.0), ("c", 3.0), ("ab", 4.0)]
            .iter()
            .map(|(s, f)| (s.as_bytes().to_vec(), *f))
            .collect();

        let model = Unigram::from(vocab, false);
        let ids = model.encode("abc").unwrap();
        assert_eq!(ids, vec![3, 2]);
    }

    #[test]
    fn test_encode_capcode() {
        let mut vocab: Vec<ScoredToken> = [
            ("my", -1.0),
            ("D", -1.0),
            (" var", -1.0),
            ("v", -1.0),
            ("ar", -1.0),
        ]
        .iter()
        .map(|(s, f)| (s.as_bytes().to_vec(), *f))
        .collect();

        let model = Unigram::from(vocab.clone(), false);
        let ids = model.encode("myvar").unwrap();
        assert_eq!(ids, vec![0, 3, 4]);

        let model = Unigram::from(vocab.clone(), true);
        let ids = model.encode("myvar").unwrap();
        assert_eq!(ids, vec![0, 1, 2]);

        vocab.push(("D var".as_bytes().to_vec(), -1.0));
        let model = Unigram::from(vocab, true);
        let ids = model.encode("myvar").unwrap();
        assert_eq!(ids, vec![0, 5]);
    }

    #[test]
    fn test_decode_encode_invariants() {
        let vocab = (0..255_u8).map(|b| (vec![b], 1.0)).collect();
        let model = Unigram::from(vocab, false);
        let input = "你好，我叫罗杰斯";

        let ids = model.encode(input).unwrap();
        assert_eq!(ids.len(), input.len());
        let decoded = model.decode(&ids).unwrap();
        assert_eq!(decoded, input);
    }
}
