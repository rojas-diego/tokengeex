use std::collections::HashSet;

use tokengeex::{Model, TokenID};

pub struct VocabularyFilter {
    vocab_size: usize,
    ids: HashSet<TokenID>,
    min_score: Option<f64>,
    force: bool,
}

impl VocabularyFilter {
    pub fn new(
        vocab_size: usize,
        ids: &[TokenID],
        min_score: Option<f64>,
        force: bool,
    ) -> VocabularyFilter {
        VocabularyFilter {
            vocab_size,
            ids: HashSet::from_iter(ids.iter().cloned()),
            min_score,
            force,
        }
    }

    /// Removes from the vocabulary tokens whose log probability is below
    /// min_score or that match ids. Does not remove past vocab_size.
    pub fn filter(&self, model: &mut Model) {
        let mut vocab = model.vocab().to_vec();
        let mut new_vocab = Vec::new();
        let mut removed = 0;

        vocab.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        for (i, token) in vocab.iter().enumerate() {
            if model.vocab_size() - removed < self.vocab_size
                || (token.keep && !self.force)
                || !self.ids.contains(&(i as TokenID))
                || token.score > self.min_score.unwrap_or(f64::NEG_INFINITY)
            {
                new_vocab.push(token.clone());
            } else {
                removed += 1;
            }
        }
        *model = Model::from(new_vocab);
    }
}
