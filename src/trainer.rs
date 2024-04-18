use super::{ScoredToken, Unigram};
use crate::{
    lattice::VecPool,
    task::{par_chunk_size, Task},
    unigram::Vocab,
    utils::lattice::Lattice,
    Model, Result, Token, TokenID, MAX_TOKEN_LENGTH,
};
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use std::{collections::HashSet, sync::RwLock};

pub struct UnigramTrainer {
    vocab_size: usize,
    num_sub_iterations: usize,
    shrinking_factor: f64,
    #[allow(dead_code)]
    sample_regularization: SampleRegularization,
}

impl UnigramTrainer {
    pub fn new(
        vocab_size: usize,
        num_sub_iterations: usize,
        shrinking_factor: f64,
        sample_regularization: SampleRegularization,
    ) -> Self {
        Self {
            vocab_size,
            num_sub_iterations,
            shrinking_factor,
            sample_regularization,
        }
    }

    /// Train a Unigram model over a dataset of sentences using the initial
    /// specified initial vocabulary.
    pub fn train(
        &mut self,
        model: &mut Unigram,
        samples: &[&str],
        keep: &HashSet<Token>,
    ) -> Result<bool> {
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
            log::info!(
                "Finalizing model | vocab_size={} desired_vocab_size={}",
                model.vocab_size(),
                desired_vocab_size
            );

            *model = self.finalize(model);

            Ok(false)
        } else {
            // Prunes pieces.
            log::info!("Pruning model | vocab_size={}", model.vocab_size());

            *model = Unigram::from(self.prune_vocab(model, model.vocab(), samples, keep)?);

            Ok(true)
        }
    }

    /// Runs the E-step of the EM algorithm for the Unigram model. It computes
    /// the expected frequencies of each token in the vocabulary. The expected
    /// frequency for each token is calculated by considering all possible
    /// segmentations of each sample into tokens, weighted by the probability
    /// of each segmentation.
    fn run_e_step(&self, model: &Unigram, samples: &[&str]) -> Vec<f64> {
        let acc_expected_frequencies = RwLock::new(vec![0.0; model.vocab.len()]);
        let chunk_size = par_chunk_size(samples.len(), 1024, 10);

        let task = Task::new("E-step", samples.len(), chunk_size);

        samples.par_chunks(chunk_size).for_each(|chunk| {
            // For each sample, we iterate over snippets of max
            // `MAX_SAMPLE_LENGTH` bytes.
            const MAX_SAMPLE_LENGTH: usize = 8192 * 10;

            let mut ltask = task.local(chunk.len());
            let mut expected_frequencies = vec![0.0; model.vocab.len()];
            let mut lattice = Lattice::default();
            let mut pool = VecPool::with_capacity(MAX_SAMPLE_LENGTH, 16);

            for sample in chunk {
                for snippet in sample.as_bytes().chunks(MAX_SAMPLE_LENGTH) {
                    debug_assert!(!sample.is_empty(), "empty sample");

                    lattice.from(
                        snippet,
                        (model.vocab.len() + 1) as TokenID,
                        (model.vocab.len() + 2) as TokenID,
                        &mut pool,
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

                ltask.record(sample.len());
            }

            // Merge the expected frequencies to the global expected frequencies
            // through a mutex.
            {
                let mut acc_expected_frequencies = acc_expected_frequencies.write().unwrap();
                for (acc, freq) in acc_expected_frequencies
                    .iter_mut()
                    .zip(expected_frequencies)
                {
                    *acc += freq;
                }
            }

            ltask.finish();
        });

        task.finish();

        acc_expected_frequencies.into_inner().unwrap()
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

        const EXPECTED_FREQUENCY_THRESHOLD: f64 = 0.5;

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
    ) -> Result<Vec<ScoredToken>> {
        let desired_vocab_size: usize = (self.vocab_size * 11) / 10; // * 1.1
        let pruned_size: usize = ((vocab.len() as f64) * self.shrinking_factor) as usize;
        let pruned_size = desired_vocab_size.max(pruned_size);
        let bos_id = (vocab.len() + 1) as TokenID;
        let eos_id = (vocab.len() + 2) as TokenID;

        // Segment each token in the vocabulary to understand how it would be
        // resegmented if it was removed from the vocabulary.
        let mut always_keep = vec![true; vocab.len()];
        let mut alternatives: Vec<Vec<TokenID>> = vec![Vec::new(); vocab.len()];
        let mut lattice = Lattice::default();
        let mut pool = VecPool::with_capacity(MAX_TOKEN_LENGTH, 16);
        for (id, (token, _)) in vocab.iter().enumerate() {
            lattice.from(token, bos_id, eos_id, &mut pool);
            model.populate_nodes(&mut lattice);

            // The second best path is the alternative to the best path. The
            // first path will always be the token itself.
            let nbests = lattice.nbest(2);

            if nbests.len() > 1 && nbests[0].len() > 1 {
                // This token is not even the first choice when segmenting
                // itself.
                always_keep[id] = false;
            }

            if nbests.len() > 1 && nbests[0].len() == 1 {
                for node in &nbests[1] {
                    let alt_id = node.token_id;
                    alternatives[id].push(alt_id);
                }
            }
        }

        let acc_frequencies = RwLock::new(vec![0usize; model.vocab.len()]);
        let chunk_size = par_chunk_size(samples.len(), 1024, 4);
        let task = Task::new("Computing frequencies", samples.len(), chunk_size);

        samples
            .par_chunks(chunk_size)
            .map(|chunk| -> Result<()> {
                let mut ltask = task.local(chunk.len());
                let mut frequencies: Vec<usize> = vec![0; vocab.len()];

                for sample in chunk {
                    for id in model.encode(sample)? {
                        frequencies[id as usize] += 1;
                    }

                    ltask.record(sample.len());
                }

                // Merge the frequencies to the global frequencies through a mutex.
                {
                    let mut acc_frequencies = acc_frequencies.write().unwrap();
                    for (acc, freq) in acc_frequencies.iter_mut().zip(frequencies) {
                        *acc += freq;
                    }
                }

                ltask.finish();

                Ok(())
            })
            .collect::<Result<()>>()?;

        task.finish();

        let token_frequencies = acc_frequencies.into_inner().unwrap();
        let sum_token_frequencies = token_frequencies.iter().sum::<usize>() as f64;
        let logsum_token_frequencies = (sum_token_frequencies as f64).ln();

        let mut candidates: Vec<(usize, f64)> = vec![];
        let mut pruned_vocab: Vec<ScoredToken> = Vec::with_capacity(pruned_size);

        log::info!("Compute model loss based on the frequencies");

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

            if token_frequencies[id] == 0 && !always_keep[id] {
                // This token never occurs?
                continue;
            } else if alternatives[id].is_empty() {
                // No alternatives. Keeps this entry.
                pruned_vocab.push((token.clone(), *score));
            } else if token_frequencies[id] != 0 {
                let freq = token_frequencies[id] as f64;
                let logprob = freq.ln() - logsum_token_frequencies;

                // After removing token `i`, its frequency is re-assigned
                // to its alternatives.
                let alt_logsum =
                    (sum_token_frequencies + freq * (alternatives.len() - 1) as f64).ln();

                // The frequencies of alternatives are increased by the
                // frequency of the removed token.
                let mut alt_logprob = 0.0;
                for &alt_id in &alternatives[id] {
                    alt_logprob +=
                        ((token_frequencies[alt_id as usize] as f64) + freq).ln() - alt_logsum;
                }

                // The difference in likelihood after removing the token.
                let loss = (freq / samples.len() as f64) * (logprob - alt_logprob);
                if !loss.is_normal() {
                    panic!(
                        "loss is f64::NaN (loss={}, freq={}, logprob={}, alt_logprob={})",
                        loss, freq, logprob, alt_logprob
                    );
                }

                candidates.push((id, loss));
            }
        }

        log::info!(
            "Updating vocabulary vocab_size={} pruned_vocab_size={} kept={}",
            vocab.len(),
            pruned_size,
            pruned_vocab.len()
        );

        candidates.sort_unstable_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        for (id, _) in candidates {
            if pruned_vocab.len() == pruned_size {
                break;
            }
            pruned_vocab.push(vocab[id].clone());
        }

        Ok(pruned_vocab)
    }

    fn finalize(&self, model: &Unigram) -> Unigram {
        let mut vocab: Vec<ScoredToken> = vec![];

        for (token, score) in model.vocab() {
            vocab.push((token.clone(), if !score.is_normal() { 0.0 } else { *score }));

            if vocab.len() >= self.vocab_size {
                break;
            }
        }

        vocab.sort_unstable_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_digamma() {
        let sum = 111111.0;
        let logsum = digamma(sum);

        println!("{}", digamma(1.0) - logsum);
        println!("{}", digamma(10.0) - logsum);
        println!("{}", digamma(100.0) - logsum);
        println!("{}", digamma(1000.0) - logsum);
        println!("{}", digamma(10000.0) - logsum);
        println!("{}", digamma(100000.0) - logsum);
    }
}
