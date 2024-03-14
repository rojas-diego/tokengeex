use std::collections::HashSet;

use super::{ScoredToken, Unigram};
use crate::{
    unigram::Vocab,
    utils::{
        lattice::Lattice,
        parallelism::{current_num_threads, MaybeParallelSlice},
    },
    Model, Token,
};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, UnigramTrainerError>;

#[derive(Debug, Error)]
pub enum UnigramTrainerError {}

pub struct UnigramTrainer {
    vocab_size: usize,
    num_sub_iterations: usize,
    shrinking_factor: f64,
}

impl UnigramTrainer {
    pub fn new(vocab_size: usize, num_sub_iterations: usize, shrinking_factor: f64) -> Self {
        Self {
            vocab_size,
            num_sub_iterations,
            shrinking_factor,
        }
    }

    /// Train a Unigram model over a dataset of sentences using the initial
    /// specified initial vocabulary.
    pub fn train(&mut self, model: &mut Unigram, samples: &[&str], keep: &HashSet<Token>) -> bool {
        let desired_vocab_size: usize = (self.vocab_size * 11) / 10;

        for i in 0..self.num_sub_iterations {
            // Compute the expected frequencies of each token. That is, how
            // much we expect to see each token in the dataset given our
            // current model.
            let expected_frequencies = self.run_e_step(model, samples);
            log::info!(
                "E-step completed subiter={} vocab_size={}",
                i,
                model.vocab_size()
            );

            // Using this expectation, we compute an alternative vocabulary
            // that maximizes the likelihood of the dataset.
            let vocab = self.run_m_step(model.vocab(), &expected_frequencies, keep);
            log::info!(
                "M-step completed subiter={} vocab_size={} alternative_vocab_size={}",
                i,
                model.vocab_size(),
                vocab.len()
            );

            *model = Unigram::from(vocab);
        }

        if model.vocab_size() <= desired_vocab_size {
            // Finally, adjusts the size of sentencepices to be |vocab_size|.
            *model = self.finalize(model);

            return false;
        }

        // Prunes pieces.
        *model = Unigram::from(self.prune_vocab(model, model.vocab(), samples, keep));

        true
    }

    /// Runs the E-step of the EM algorithm for the Unigram model. It computes
    /// the expected frequencies of each token in the vocabulary. The expected
    /// frequency for each token is calculated by considering all possible
    /// segmentations of each sample into tokens, weighted by the probability
    /// of each segmentation.
    fn run_e_step(&self, model: &Unigram, samples: &[&str]) -> Vec<f64> {
        let chunk_size = std::cmp::max(samples.len() / current_num_threads(), 1);
        let expected_frequencies: Vec<f64> = samples
            .maybe_par_chunks(chunk_size)
            .map(|chunk| {
                // How much we expect to see each token in the vocabulary
                // in this chunk of samples. This frequency is computed based
                // on the probability of the token being part of optimal splits
                // (its marginal probability).
                let mut expected_frequencies: Vec<f64> = vec![0.0; model.vocab_size()];

                for &sample in chunk {
                    debug_assert!(!sample.is_empty(), "empty sample");

                    // Compute all the possible segmentations of the sample.
                    let mut lattice = Lattice::from(
                        sample.as_bytes(),
                        model.vocab.len() + 1,
                        model.vocab.len() + 2,
                    );
                    model.populate_nodes(&mut lattice);

                    let z = lattice.populate_marginal(&mut expected_frequencies);
                    if !z.is_normal() {
                        panic!(
                            "normalization constant is f64::NaN (z={}, len={})",
                            z,
                            sample.len()
                        );
                    }
                }

                expected_frequencies
            })
            .reduce(
                || vec![0.0; model.vocab_size()],
                |l, r| l.iter().zip(r).map(|(a, b)| a + b).collect(),
            );

        expected_frequencies
    }

    /// Runs the M-step of the EM algorithm for the Unigram model. It computes
    /// an alternative vocabulary based on the expected frequencies.
    fn run_m_step(
        &self,
        vocab: &[ScoredToken],
        expected_frequencies: &[f64],
        keep: &HashSet<Token>,
    ) -> Vec<ScoredToken> {
        assert_eq!(vocab.len(), expected_frequencies.len());

        const EXPECTED_FREQUENCY_THRESHOLD: f64 = 0.1;

        let mut alternative_vocab: Vec<ScoredToken> = Vec::with_capacity(self.vocab_size);

        for (freq, (token, _)) in expected_frequencies.iter().zip(vocab) {
            if *freq < EXPECTED_FREQUENCY_THRESHOLD && !keep.contains(token) {
                continue;
            }

            alternative_vocab.push((token.clone(), f64::max(*freq, EXPECTED_FREQUENCY_THRESHOLD)));
        }

        // I have no clue what's going here. A previous comment pointed to this
        // paper:
        // https://cs.stanford.edu/~pliang/papers/tutorial-acl2007-talk.pdf
        let sum_token_frequencies = alternative_vocab.iter().map(|(_, c)| *c).sum::<f64>();
        let logsum_token_frequencies = digamma(sum_token_frequencies);
        let scores: Vec<f64> = alternative_vocab
            .iter()
            .map(|(_, score)| digamma(*score) - logsum_token_frequencies)
            .collect();

        // Ensure the vocabulary does not contain f64::NaN.
        for (i, freq) in scores.iter().enumerate() {
            if freq.is_nan() || freq.is_infinite() {
                let token = alternative_vocab[i].0.clone();
                let freq = alternative_vocab[i].1;

                // Write the alternative vocabulary to a file for debugging.
                let vocab_len = alternative_vocab.len();
                let vocab = Vocab::from(alternative_vocab);

                std::fs::write(
                    format!("tokengeex-dump-vocab-{}.json", vocab_len),
                    serde_json::to_string(&vocab).unwrap(),
                )
                .unwrap();

                panic!(
                    "M-step: alternative vocabulary contains invalid frequency for token {:?}, {:?}: {}",
                    token, String::from_utf8_lossy(&token), freq
                );
            }
        }

        alternative_vocab
            .into_iter()
            .zip(scores)
            .map(|((token, _), score)| (token, score))
            .collect()
    }

    /// This method returns a new vocabulary with the least useful tokens
    /// removed.
    fn prune_vocab(
        &self,
        model: &Unigram,
        vocab: &[ScoredToken],
        samples: &[&str],
        keep: &HashSet<Token>,
    ) -> Vec<ScoredToken> {
        let bos_id = vocab.len() + 1;
        let eos_id = vocab.len() + 2;

        // Segment each token in the vocabulary to understand how it would be
        // resegmented if it was removed from the vocabulary.
        let mut has_alternative = vec![true; vocab.len()];
        let mut alternatives: Vec<Vec<usize>> = vec![Vec::new(); vocab.len()];
        for (id, (token, _)) in vocab.iter().enumerate() {
            let mut lattice = Lattice::from(token, bos_id, eos_id);
            model.populate_nodes(&mut lattice);

            // The second best path is the alternative to the best path. The
            // first path will always be the token itself.
            let nbests = lattice.nbest(2);

            if nbests.len() == 1 {
                // There is no other way to segment this token. Keep it.
                has_alternative[id] = true;
            } else if nbests[0].len() > 1 {
                // Does that mean that the token's score is so low that Unigram
                // considers segmenting itself using two other tokens?
                has_alternative[id] = false;
            } else if nbests[0].len() == 1 {
                has_alternative[id] = true;
                for node in &nbests[1] {
                    let alt_id = node.borrow().id;
                    alternatives[id].push(alt_id);
                }
            }
        }

        // For a token ID i, inverted[i] stores the list of sentence IDs that
        // contain the token i.
        // This step computes the global frequency of each token and the list of
        // samples that contain each token.
        let chunk_size = std::cmp::max(samples.len() / current_num_threads(), 1);
        let token_frequencies: Vec<usize> = samples
            .maybe_par_chunks(chunk_size)
            .map(|chunk| {
                let mut freq: Vec<usize> = vec![0; vocab.len()];

                for sentence in chunk {
                    for id in model.encode(sentence) {
                        freq[id as usize] += 1;
                    }
                }

                freq
            })
            .reduce(
                || vec![0; vocab.len()],
                |l, r| l.iter().zip(r).map(|(a, b)| a + b).collect(),
            );

        let sum_token_frequencies = token_frequencies.iter().sum::<usize>() as f64;
        let logsum_token_frequencies = (sum_token_frequencies as f64).ln();

        let mut candidates: Vec<(usize, f64)> = vec![];
        let mut pruned_vocab: Vec<ScoredToken> = Vec::with_capacity(self.vocab_size);

        // Compute how likely the LM likelihood is reduced if the token `i` is
        // removed from the vocabulary.
        // Since the exact computation of loss is difficult, we compute the loss
        // approximately by assuming that all tokens `i` would be replaced with
        // alternatives[i] if removed.
        for (id, (token, score)) in vocab.iter().enumerate() {
            if keep.contains(token) {
                pruned_vocab.push((token.clone(), *score));
                continue;
            }

            if token_frequencies[id] == 0 && !has_alternative[id] {
                // This token never occurs?
                continue;
            } else if alternatives[id].is_empty() {
                // No alternatives. Keeps this entry.
                pruned_vocab.push((token.clone(), *score));
            } else if token_frequencies[id] != 0 {
                let freq = token_frequencies[id] as f64;
                let logprob = freq.ln() - logsum_token_frequencies;

                // After removing token `i`, its frequency is re-assigned
                // its alternatives.
                let logsum_alternative =
                    (sum_token_frequencies + freq * (alternatives.len() - 1) as f64).ln();

                // The frequencies of alternatives are increased by the
                // frequency of the removed token.
                let mut logprob_alternative = 0.0;
                for n in &alternatives[id] {
                    logprob_alternative +=
                        ((token_frequencies[*n] as f64) + freq).ln() - logsum_alternative;
                }

                // The difference in likelihood after removing the token.
                let loss = (freq / samples.len() as f64) * (logprob - logprob_alternative);
                if !loss.is_normal() {
                    panic!(
                        "loss is f64::NaN (loss={}, freq={}, logprob={}, logprob_alternative={})",
                        loss, freq, logprob, logprob_alternative
                    );
                }

                candidates.push((id, loss));
            }
        }

        let desired_vocab_size: usize = (self.vocab_size * 11) / 10; // * 1.1
        let pruned_size: usize = ((vocab.len() as f64) * self.shrinking_factor) as usize;
        let pruned_size = desired_vocab_size.max(pruned_size);

        candidates.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        for (id, _) in candidates {
            if pruned_vocab.len() == pruned_size {
                break;
            }
            pruned_vocab.push(vocab[id].clone());
        }

        pruned_vocab.to_vec()
    }

    fn finalize(&self, model: &Unigram) -> Unigram {
        let mut vocab: Vec<ScoredToken> = vec![];

        for (token, score) in model.vocab() {
            vocab.push((token.clone(), if !score.is_normal() { 0.0 } else { *score }));

            if vocab.len() >= self.vocab_size {
                break;
            }
        }

        vocab.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        Unigram::from(vocab)
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
