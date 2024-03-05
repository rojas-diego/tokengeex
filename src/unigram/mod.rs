// Code imported and modified from: https://github.com/huggingface/tokenizers
// License: https://github.com/huggingface/tokenizers/blob/4a8105c36671ef46738d6e2799c55198139b87b2/LICENSE

use crate::{
    core::Model,
    utils::{
        lattice::Lattice,
        trie::{Trie, TrieBuilder},
    },
};
use std::collections::HashMap;

mod serialization;
mod trainer;

pub use serialization::Vocab;
pub use trainer::*;

/// An arbitrary sequence of bytes. Almost always valid UTF-8 but not
/// guaranteed.
pub type Token = Vec<u8>;

/// The byte fallbacks for the first 256 ASCII characters.
pub type ScoredToken = (Token, f64);

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
    trie: Trie<(usize, usize)>,
}

impl Unigram {
    /// Create a new `Unigram` model from a vocabulary of scored tokens.
    // TODO: Should take a UNK token as an argument.
    pub fn from(vocab: Vec<ScoredToken>) -> Self {
        let mut token_to_ids: HashMap<Token, u32> = HashMap::new();
        let mut trie_builder = TrieBuilder::default();

        for (id, (token, _)) in vocab.iter().enumerate() {
            token_to_ids.insert(token.clone(), id as u32);
            trie_builder.push(token, (id, token.len()));
        }

        let trie = trie_builder.build();

        Self {
            vocab,
            token_to_ids,
            trie,
        }
    }

    /// Populates a lattice with all the possible tokenizations of the input
    /// sentence.
    // TODO: At the moment, if there's no way to tokenize the sentence, we
    // panic. We should use an UNK token instead.
    pub(super) fn populate_nodes(&self, lattice: &mut Lattice) {
        let input = lattice.sentence;

        for pos in 0..input.len() {
            let suffix = &input[pos..];

            for (id, len) in self.trie.common_prefix_search(suffix.iter().copied()) {
                let score = &self.vocab[id].1;

                lattice.insert(pos, len, *score, id);
            }
        }
    }

    /// Iterate of vocabulary of the model as a pair of `(token, score)`.
    pub(super) fn iter(&self) -> UnigramIterator {
        UnigramIterator { model: self, i: 0 }
    }

    /// Access the vocabulary of the model.
    pub fn vocab(&self) -> &Vec<ScoredToken> {
        &self.vocab
    }
}

/// Iterator to iterate of vocabulary of the model, and their relative score.
pub struct UnigramIterator<'a> {
    model: &'a Unigram,
    i: usize,
}

impl<'a> Iterator for UnigramIterator<'a> {
    type Item = &'a ScoredToken;

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
    /// Encode the input sequence into a sequence of token IDs in O(n) time
    /// using the SentencePiece DP algorithm.
    // TODO: At the moment, if there's no way to tokenize the sentence, we
    // panic. We should use an UNK token instead.
    fn encode(&self, input: &str) -> Vec<u32> {
        let input = input.as_bytes();

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

        for pos in 0..input.len() {
            // We skip positions that are unreachable.
            if dp[pos].start.is_none() {
                continue;
            }

            let suffix = &input[pos..];

            for (id, len) in self.trie.common_prefix_search(suffix.iter().copied()) {
                let node = &dp[pos + len];
                let score = dp[pos].score + self.vocab[id].1;

                if node.start.is_none() || score > node.score {
                    dp[pos + len] = Node {
                        id: id as u32,
                        score,
                        start: Some(pos),
                    };
                }
            }
        }

        // Backtrack along the best path to recover the tokens.
        let mut pos = input.len();
        let mut ids: Vec<u32> = vec![];

        while pos > 0 {
            let node = &dp[pos];

            let start = node.start.unwrap_or_else(|| {
                panic!(
                    "encode: current node at pos {}/{} (id={}, score={}) has no start position",
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

    /// Decode the input sequence of token IDs into a string in O(n) time. If
    /// the string is not valid UTF-8, it will be returned as a lossy string.
    ///
    /// # Panics
    ///
    /// This method will panic if any of the token IDs are out of bounds.
    fn decode(&self, ids: &[u32]) -> String {
        let mut res = Vec::new();

        for &id in ids {
            if id >= self.vocab_size() as u32 {
                panic!(
                    "decode: token ID {} is out of bounds (vocab size is {})",
                    id,
                    self.vocab.len()
                );
            }

            let (token, _) = &self.vocab[id as usize];

            res.extend_from_slice(token);
        }

        String::from_utf8_lossy(&res).into_owned()
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

        let model = Unigram::from(vocab);
        let ids = model.encode("abc");
        assert_eq!(ids, vec![3, 2]);
    }

    #[test]
    fn test_decode_encode_invariants() {
        let vocab = (0..255_u8).map(|b| (vec![b], 1.0)).collect();
        let model = Unigram::from(vocab);
        let input = "你好，我叫罗杰斯";

        let ids = model.encode(input);
        assert_eq!(ids.len(), input.len());
        let decoded = model.decode(&ids);
        assert_eq!(decoded, input);
    }
}
