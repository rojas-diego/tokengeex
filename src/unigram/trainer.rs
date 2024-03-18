use super::{ScoredToken, Unigram};
use crate::{
    lattice::VecPool,
    unigram::Vocab,
    utils::{
        lattice::Lattice,
        parallelism::{current_num_threads, MaybeParallelSlice},
    },
    Model, Result, Token, TokenID, MAX_TOKEN_LENGTH,
};
use std::{collections::HashSet, sync::atomic::Ordering::Relaxed, sync::RwLock, thread::ThreadId};

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
            *model = self.finalize(model);

            Ok(false)
        } else {
            // Prunes pieces.
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
        // We chunk up samples into chunks that are at most 1/10th of the
        // per-thread workload because too large chunks can cause some threads
        // to be idle while others are still working. We also prevent
        // chunks from being too small to avoid too much overhead.
        let min_chunk_size = 1024;
        let max_chunk_size = samples.len() / current_num_threads() / 10;
        let chunk_size = std::cmp::max(
            1,
            if max_chunk_size < min_chunk_size {
                // If we have a small amount of samples, it's better to have
                // exactly num_threads chunks.
                ((samples.len() as f64) / (current_num_threads() as f64)).ceil() as usize
            } else {
                max_chunk_size
            },
        );
        // At the end of each chunk, each thread merges its expected
        // frequencies to the global expected frequencies through a mutex.
        let acc_expected_frequencies = RwLock::new(vec![0.0; model.vocab.len()]);

        log::info!(
            "E-step | {} samples | {} threads | {} chunks | {} chunk size",
            samples.len(),
            current_num_threads(),
            (samples.len() + chunk_size) / chunk_size,
            chunk_size
        );

        let e_step_start = std::time::Instant::now();
        let num_samples = samples.len();
        let num_samples_processed = std::sync::atomic::AtomicUsize::new(0);

        samples.maybe_par_chunks(chunk_size).for_each(|chunk| {
            // For each sample, we iterate over snippets of max
            // `MAX_SAMPLE_LENGTH` bytes.
            const MAX_SAMPLE_LENGTH: usize = 8192;

            let tid =
                unsafe { std::mem::transmute::<ThreadId, usize>(std::thread::current().id()) };
            let start = std::time::Instant::now();
            let mut processed_bytes = 0;
            let mut expected_frequencies = vec![0.0; model.vocab.len()];
            let mut lattice = Lattice::default();
            let mut pool = VecPool::with_capacity(MAX_SAMPLE_LENGTH, 16);

            fn mb_per_sec(n: usize, since: std::time::Instant) -> f64 {
                (n as f64 / 1024.0 / 1024.0) / since.elapsed().as_secs_f64()
            }

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

                processed_bytes += sample.len();
                num_samples_processed.fetch_add(1, Relaxed);
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

            let total_samples_processed = num_samples_processed.load(Relaxed);
            let percent_done = (total_samples_processed as f64 / num_samples as f64) * 100.0;
            let eta =
                (e_step_start.elapsed().as_secs_f64() / percent_done) * (100.0 - percent_done);

            log::debug!(
                "Worker {:>3} | ETA {:>5}s | {:>6.2}% | {:>5} samples in {:>3}s | {:>4.2}MB/s",
                tid,
                eta.round(),
                percent_done,
                chunk.len(),
                start.elapsed().as_secs_f64() as usize,
                mb_per_sec(processed_bytes, start),
            );
        });

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
    ) -> Result<Vec<ScoredToken>> {
        let bos_id = (vocab.len() + 1) as TokenID;
        let eos_id = (vocab.len() + 2) as TokenID;

        // Segment each token in the vocabulary to understand how it would be
        // resegmented if it was removed from the vocabulary.
        let mut has_alternative = vec![true; vocab.len()];
        let mut alternatives: Vec<Vec<TokenID>> = vec![Vec::new(); vocab.len()];
        let mut lattice = Lattice::default();
        let mut pool = VecPool::with_capacity(MAX_TOKEN_LENGTH, 16);
        for (id, (token, _)) in vocab.iter().enumerate() {
            lattice.from(token, bos_id, eos_id, &mut pool);
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
                    let alt_id = node.token_id;
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
                    for id in model.encode(sentence)? {
                        freq[id as usize] += 1;
                    }
                }

                Ok(freq)
            })
            .reduce(
                || Ok(vec![0; vocab.len()]),
                |l: Result<Vec<usize>>, r: Result<Vec<usize>>| {
                    let l = l?;
                    let r = r?;
                    Ok(l.iter().zip(r).map(|(a, b)| a + b).collect())
                },
            )?;

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
                        ((token_frequencies[*n as usize] as f64) + freq).ln() - logsum_alternative;
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

        Ok(pruned_vocab.to_vec())
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
