use tokengeex::Model;

pub struct VocabularyFilter {
    vocab_size: usize,
    min_score: Option<f64>,
    force: bool,
}

impl VocabularyFilter {
    pub fn new(vocab_size: usize, min_score: Option<f64>, force: bool) -> VocabularyFilter {
        VocabularyFilter {
            vocab_size,
            min_score,
            force,
        }
    }

    /// Removes from the vocabulary tokens whose log probability is below
    /// min_score or that match ids. Does not remove past vocab_size.
    pub fn filter(&self, model: &mut Model) {
        if model.vocab_size() <= self.vocab_size {
            return;
        }

        let num_tokens_to_remove = model.vocab_size() - self.vocab_size;
        let mut num_tokens_removed = 0;
        let mut vocab = model.vocab().to_vec();
        let mut new_vocab = Vec::new();

        vocab.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        for token in vocab.iter() {
            let should_keep = num_tokens_removed >= num_tokens_to_remove
                || (token.keep && !self.force)
                || token.score > self.min_score.unwrap_or(f64::NEG_INFINITY);

            if should_keep {
                new_vocab.push(token.clone());
            } else {
                num_tokens_removed += 1;
            }
        }

        *model = Model::from(new_vocab);
    }
}
