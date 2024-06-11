// Code imported and modified from: https://github.com/huggingface/tokenizers
// License: https://github.com/huggingface/tokenizers/blob/4a8105c36671ef46738d6e2799c55198139b87b2/LICENSE

use crate::{lattice::Lattice, trie::Trie, Error, Result, ScoredToken, Token, TokenID};
use std::collections::HashMap;

#[derive(Clone, Default)]
pub struct Model {
    vocab: Vec<ScoredToken>,
    token_to_ids: HashMap<Token, u32>,
    trie: Trie<(TokenID, u32)>,
}

impl Model {
    /// Create a new `Unigram` model from a vocabulary of scored tokens.
    pub fn from(vocab: Vec<ScoredToken>) -> Self {
        let mut token_to_ids: HashMap<Token, u32> = HashMap::new();
        let mut trie = Trie::default();

        for (id, token) in vocab.iter().enumerate() {
            token_to_ids.insert(token.value.clone(), id as u32);
            trie.push(&token.value, (id as u32, token.len() as u32));
        }

        Self {
            vocab,
            token_to_ids,
            trie,
        }
    }

    /// Populates a lattice with all the possible tokenizations of the input
    /// sentence.
    pub fn populate_nodes(&self, lattice: &mut Lattice, dropout: f64) {
        let mut buff = Vec::<u8>::with_capacity(256);
        let input = lattice.sentence;

        for pos in 0..input.len() {
            let suffix = &input[pos..];

            buff.clear();
            for (id, len) in self
                .trie
                .common_prefix_search(suffix.iter().copied(), &mut buff)
            {
                let score = &self.vocab[id as usize].score;

                if len > 1 && dropout > 0.0 && rand::random::<f64>() < dropout {
                    continue;
                }

                lattice.insert(pos, id, len as usize, *score);
            }
        }
    }

    /// Encode the input sequence into a sequence of token IDs in O(n) time
    /// using the SentencePiece DP algorithm.
    pub fn encode(&self, input: &str, dropout: f64) -> Result<Vec<u32>> {
        let mut buff = Vec::<u8>::with_capacity(256);
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
                start: None,
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

            buff.clear();
            for (id, len) in self
                .trie
                .common_prefix_search(suffix.iter().copied(), &mut buff)
            {
                let len = len as usize;
                let node = &dp[pos + len];
                let score = dp[pos].score + self.vocab[id as usize].score;

                if (dropout <= 0.0 || len <= 1 || dropout < rand::random::<f64>())
                    && (node.start.is_none() || score > node.score)
                {
                    dp[pos + len] = Node {
                        id,
                        score,
                        start: Some(pos),
                    };
                }
            }
        }

        // Backtrack along the best path to recover the tokens.
        let mut pos = input.len();
        let mut ids: Vec<u32> = Vec::with_capacity(input.len() / 2);

        while pos > 0 {
            let node = &dp[pos];

            let start = node.start.ok_or_else(|| Error::NoPath(pos, input.len()))?;

            ids.push(node.id);
            pos = start;
        }

        // Reverse to maintain original order since we built it backwards.
        ids.reverse();

        Ok(ids)
    }

    /// Efficiently search for any token that is a prefix of `s`.
    pub fn common_prefix_search<'a>(
        &'a self,
        s: &'a [u8],
        buffer: &'a mut Vec<u8>,
    ) -> impl Iterator<Item = (TokenID, u32)> + 'a {
        self.trie.common_prefix_search(s.iter().copied(), buffer)
    }

    /// Decode the input sequence of token IDs into a string in O(n) time. If
    /// the string is not valid UTF-8, it will be returned as a lossy string.
    ///
    /// # Panics
    ///
    /// This method will panic if any of the token IDs are out of bounds.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut res = Vec::new();

        for &id in ids {
            if id >= self.vocab_size() as u32 {
                return Err(Error::TokenIdOutOfBounds(id));
            }

            let token = &self.vocab[id as usize];

            res.extend_from_slice(&token.value);
        }

        Ok(String::from_utf8_lossy(&res).into_owned())
    }

    /// Convert a token to a token ID. Currently it is not possible to access
    /// tokens that are invalid UTF-8 through this method.
    pub fn token_to_id(&self, token: &Token) -> Option<u32> {
        self.token_to_ids.get(token).copied()
    }

    /// Convert a token ID to a token. If the byte sequence is not valid UTF-8
    /// it will be returned as a lossy string.
    pub fn id_to_token(&self, id: u32) -> Option<ScoredToken> {
        if id >= self.vocab.len() as u32 {
            return None;
        }

        Some(self.vocab[id as usize].clone())
    }

    /// Number of entries in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Add a token to the vocabulary.
    pub fn add_tokens<I>(&mut self, tokens: I)
    where
        I: IntoIterator<Item = ScoredToken>,
    {
        for token in tokens {
            let id = self.vocab.len() as u32;
            self.trie.push(&token.value, (id, token.len() as u32));
            self.token_to_ids.insert(token.value.clone(), id);
            self.vocab.push(token);
        }
    }

    /// Access the vocabulary of the model.
    pub fn vocab(&self) -> &[ScoredToken] {
        &self.vocab
    }
}

#[cfg(test)]
mod tests {
    use crate::{make_vocab, new_default_vocab};

    use super::*;

    #[test]
    fn test_encode() {
        let vocab = make_vocab(&[(b"a", -3.0), (b"b", -3.0), (b"c", -3.0), (b"ab", -4.0)]);

        let model = Model::from(vocab);
        let ids = model.encode("abc", 0.0).unwrap();
        assert_eq!(ids, vec![3, 2]);
    }

    #[test]
    fn test_encode_dropout() {
        let vocab = make_vocab(&[
            (b"a", -3.0),
            (b"b", -3.0),
            (b"c", -3.0),
            (b"d", -3.0),
            (b"e", -3.0),
            (b"f", -3.0),
            (b"ab", -4.0),
            (b"abc", -5.0),
            (b"abcd", -6.0),
            (b"abcde", -7.0),
            (b"abcdef", -8.0),
        ]);

        let model = Model::from(vocab);
        let ids = model.encode("abcdef", 1.0).unwrap();
        println!("{:?}", ids);
        assert_eq!(ids, vec![0, 1, 2, 3, 4, 5]);

        let ids = model.encode("abcdef", 0.5).unwrap();
        println!("{:?}", ids);
    }

    #[test]
    fn test_decode_encode_invariants() {
        let vocab = new_default_vocab();
        let model = Model::from(vocab);
        let input = "你好，我叫罗杰斯";

        let ids = model.encode(input, 0.0).unwrap();
        assert_eq!(ids.len(), input.len());
        let decoded = model.decode(&ids).unwrap();
        assert_eq!(decoded, input);
    }
}
