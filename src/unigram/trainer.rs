use super::{ScoredToken, Unigram};
use crate::{
    capcode,
    unigram::Vocab,
    utils::{
        lattice::Lattice,
        parallelism::{current_num_threads, MaybeParallelSlice},
    },
    Model,
};
use derive_builder::Builder;
use rand::Rng;
use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet},
};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, UnigramTrainerError>;

#[derive(Debug, Clone)]
/// Vocabulary generator is used to construct the initial vocabulary from the
/// training dataset. This might become a trait later but right now it picks
/// the most common substrings in the dataset and scores them based on their
/// frequency and length.
pub struct VocabularyGenerator {
    words_per_token: usize,
    window_size: usize,
    insert_probability: f64,
    suggested_tokens: Vec<String>,
    strict: bool,
}

impl VocabularyGenerator {
    // TODO: Need to validate that insert probability is between 0 and 1.
    pub fn new(
        words_per_token: usize,
        window_size: usize,
        insert_probability: f64,
        suggested_tokens: Vec<String>,
        strict: bool,
    ) -> Self {
        Self {
            words_per_token,
            window_size,
            insert_probability,
            suggested_tokens,
            strict,
        }
    }

    /// Generates an initial vocabulary from the given samples.
    pub fn generate_vocabulary<'a>(
        self,
        samples: impl Iterator<Item = &'a str>,
        size: usize,
    ) -> Vec<ScoredToken> {
        let mut tokens = HashMap::new();
        let mut byte_freq = [0usize; 256];
        let mut rng = rand::thread_rng();

        for sample in samples {
            let mut sample_tokens = HashSet::new();

            for (i, _) in sample.char_indices() {
                let suffix = &sample[i..];
                for (ii, c) in suffix.char_indices().skip(1).take(self.window_size) {
                    let candidate = &suffix[..ii + c.len_utf8()];

                    // TODO: Dirty fix to avoid byte fallback tokens that we
                    // add manually.
                    if candidate.len() > 1
                        && self.is_valid_token(candidate)
                        && rng.gen_range(0.0..1.0) < self.insert_probability
                    {
                        sample_tokens.insert(candidate);
                    }
                }
            }

            for b in sample.bytes() {
                if byte_freq[b as usize] == 0 || rng.gen_range(0.0..1.0) < self.insert_probability {
                    byte_freq[b as usize] += 1;
                }
            }

            for token in sample_tokens {
                *tokens.entry(token).or_insert(0) += 1;
            }
        }

        // Collect the frequency of the suggested tokens.
        // TODO: Instead of relying on a default frequency of 1, we should
        // consider the frequency of the suggested tokens in the dataset. This
        // involves making a second pass over the dataset.
        let suggested_tokens_freq = self
            .suggested_tokens
            .iter()
            .map(|token| tokens.get(token.as_str()).copied().unwrap_or(1))
            .collect::<Vec<usize>>();

        // Convert the tokens to a vector and sort them by frequency.
        log::info!("Sorting tokens by frequency");
        let mut tokens: Vec<_> = tokens.into_iter().collect();
        tokens.sort_by_key(|(_, freq)| Reverse(*freq));

        // Keep track of duplicates, ensuring the earlier occurrence is kept.
        let mut seen: HashSet<&str> = HashSet::new();

        // Add all 256 ASCII characters and byte values to the initial
        // vocabulary.
        let mut vocab: Vec<ScoredToken> = (0..255_u8)
            .map(|i| (vec![i], (byte_freq[i as usize] as f64) * (i as f64)))
            .collect();

        // Add the suggested tokens.
        for (i, token) in self.suggested_tokens.iter().enumerate() {
            if !seen.contains(token.as_str()) {
                seen.insert(token);
                vocab.push((
                    token.as_bytes().to_vec(),
                    (suggested_tokens_freq[i] as f64) * (token.len() as f64),
                ));
            }
        }

        // We further add the most frequent substrings.
        for (token, freq) in tokens {
            if vocab.len() >= size {
                break;
            }

            if !seen.contains(token) {
                vocab.push((token.as_bytes().to_vec(), (freq * token.len()) as f64));
            }

            seen.insert(token);
        }

        // Sort the vocabulary by score.
        vocab.sort_by(|(_, a), (_, b)| {
            a.partial_cmp(b)
                .unwrap_or(std::cmp::Ordering::Equal)
                .reverse()
        });

        // Convert the scores to log probabilities.
        to_log_prob(&mut vocab);

        // Computing log probabilities generates NaNs for items where freq=0.
        vocab.iter_mut().for_each(|(_, score)| {
            if !score.is_normal() {
                *score = 0.0;
            }
        });

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
        let mut has_capcode = false;
        let mut num_spaces = 0;
        let mut num_words = 0; // Number of whitespace separated sequences of characters.
        let mut previous_char = ' ';
        let begins_with_space = token.starts_with(' ');

        for c in token.chars() {
            if capcode::is_marker(c) {
                has_capcode = true;
            } else if c.is_ascii() {
                has_ascii = true;

                if c.is_alphanumeric() {
                    has_word = true;

                    if !previous_char.is_alphanumeric() {
                        num_words += 1;
                    }
                } else if c == ' ' {
                    num_spaces += 1;
                } else {
                    has_punctuation = true;
                }
            } else {
                has_non_ascii = true;
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
            if has_word
                && (previous_char.is_whitespace()
                    || num_spaces > num_words
                    || has_punctuation
                    || has_capcode)
            {
                return false;
            }
            if num_words > 1 && !begins_with_space {
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
    sentences: Vec<String>,
}

impl Default for UnigramTrainer {
    fn default() -> Self {
        Self::builder().build().unwrap()
    }
}

#[derive(Debug, Error)]
pub enum UnigramTrainerError {}

impl UnigramTrainer {
    pub fn builder() -> UnigramTrainerBuilder {
        UnigramTrainerBuilder::default()
    }

    pub fn feed(&mut self, sentence: &str) {
        let mut start = 0;

        while start < sentence.len() {
            // Determine the end of the current chunk without splitting a character
            let end = if start + 32 > sentence.len() {
                sentence.len()
            } else {
                // Find the end that does not split a character
                let mut end = start + 32;
                while !sentence.is_char_boundary(end) {
                    end += 1;
                }
                end
            };

            // Push the current chunk
            self.sentences.push(sentence[start..end].to_string());

            // Update the start for the next chunk
            start = end;
        }
    }

    /// Train a Unigram model over a dataset of sentences using the initial
    /// specified initial vocabulary.
    pub fn train(&self, model: &mut Unigram, vocab: Vec<ScoredToken>) -> Result<()> {
        let desired_vocab_size: usize = (self.vocab_size * 11) / 10;
        let expected_loops = (((desired_vocab_size as f64).ln() - (vocab.len() as f64).ln())
            / self.shrinking_factor.ln()) as usize
            + 1;

        log::info!(
            "Training on {} sentences using an initial vocab of {} tokens. Running {} EM loop(s) to fine grain the vocabulary",
            self.sentences.len(),
            vocab.len(),
            expected_loops
        );

        *model = Unigram::from(vocab);

        for i in 0..usize::MAX {
            for ii in 0..self.num_sub_iterations {
                log::info!("Running E-step iter={} subiter={}", i, ii);
                // Compute the expected frequencies of each token. That is, how
                // much we expect to see each token in the dataset given our
                // current model.
                let (objective, num_tokens, num_bytes, expected_frequencies) =
                    self.run_e_step(model);

                log::info!("Running M-step iter={} subiter={}", i, ii);
                // Using this expectation, we compute an alternative vocabulary
                // that maximizes the likelihood of the dataset.
                let vocab = self.run_m_step(model.vocab(), &expected_frequencies);

                log::info!(
                    "Completed EM iter={} subiter={} vocab_size={} objective={} num_tokens={} compression={}",
                    i,
                    ii,
                    model.vocab_size(),
                    objective,
                    num_tokens,
                    (num_bytes as f64) / (num_tokens as f64)
                );

                *model = Unigram::from(vocab);
            }

            // Stops the iteration when the size of sentences reaches to the
            // desired symbol size.
            if model.vocab_size() <= desired_vocab_size {
                break;
            }

            // Prunes pieces.
            *model = Unigram::from(self.prune_vocab(model, model.vocab()));
        }

        // Finally, adjusts the size of sentencepices to be |vocab_size|.
        *model = self.finalize(model);

        Ok(())
    }

    /// Runs the E-step of the EM algorithm for the Unigram model. It computes
    /// the expected frequencies of each token in the vocabulary. The expected
    /// frequency for each token is calculated by considering all possible
    /// segmentations of each sentence into tokens, weighted by the probability
    /// of each segmentation.
    fn run_e_step(&self, model: &Unigram) -> (f64, u32, u32, Vec<f64>) {
        let chunk_size = std::cmp::max(self.sentences.len() / current_num_threads(), 1);
        let (loss, num_tokens, num_bytes, expected_frequencies): (f64, u32, u32, Vec<f64>) = self.sentences
            .maybe_par_chunks(chunk_size)
            .map(|chunk| {
                // A measure of how well the model fits the data.
                let mut loss: f64 = 0.0;
                // How much we expect to see each token in the vocabulary
                // in this chunk of sentences. This frequency is computed based
                // on the probability of the token being part of optimal splits
                // (its marginal probability).
                let mut expected_frequencies: Vec<f64> = vec![0.0; model.vocab_size()];
                let mut num_tokens: u32 = 0;
                let mut num_bytes: u32 = 0;

                for sentence in chunk {
                    // Compute all the possible segmentations of the sentence.
                    let mut lattice =
                        Lattice::from(sentence.as_bytes(), model.vocab.len() + 1, model.vocab.len() + 2);
                    model.populate_nodes(&mut lattice);

                    let z = lattice.populate_marginal(&mut expected_frequencies);
                    if !z.is_normal() {
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

                        // We log before panicking because it seems the parallelism
                        // is causing all threads to panic at the same time.
                        log::info!("normalization constant for sentence {:?} is f64::NaN lattice={} tokens={:?}", sentence, lattice, tokens);

                        panic!("normalization constant is f64::NaN");
                    }

                    num_tokens += lattice.viterbi().len() as u32;
                    num_bytes += sentence.len() as u32;
                    loss -= z / (self.sentences.len() as f64);
                }
                (loss, num_tokens, num_bytes, expected_frequencies)
            })
            .reduce(
                || (0.0, 0, 0, vec![0.0; model.vocab_size()]),
                |(loss, ntokens, nbytes, expected), (lloss, lntokens, lnbytes, lexpected)| {
                    (
                        loss + lloss,
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

        (loss, num_tokens, num_bytes, expected_frequencies)
    }

    /// Runs the M-step of the EM algorithm for the Unigram model. It computes
    /// an alternative vocabulary based on the expected frequencies.
    fn run_m_step(&self, vocab: &[ScoredToken], expected_frequencies: &[f64]) -> Vec<ScoredToken> {
        assert_eq!(vocab.len(), expected_frequencies.len());

        const EXPECTED_FREQUENCY_THRESHOLD: f64 = 0.5;

        let mut alternative_vocab: Vec<ScoredToken> = Vec::with_capacity(self.vocab_size);

        for (freq, (token, _)) in expected_frequencies.iter().zip(vocab) {
            if *freq < EXPECTED_FREQUENCY_THRESHOLD {
                continue;
            }

            alternative_vocab.push((token.clone(), *freq));
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
            if !freq.is_normal() {
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
    fn prune_vocab(&self, model: &Unigram, vocab: &[ScoredToken]) -> Vec<ScoredToken> {
        let bos_id = vocab.len() + 1;
        let eos_id = vocab.len() + 2;

        // Segment each token in the vocabulary to understand how it would be
        // resegmented if it was removed from the vocabulary.
        let mut always_keep = vec![true; vocab.len()];
        let mut alternatives: Vec<Vec<usize>> = vec![Vec::new(); vocab.len()];
        for (id, (token, _)) in vocab.iter().enumerate() {
            let mut lattice = Lattice::from(token, bos_id, eos_id);
            model.populate_nodes(&mut lattice);

            // The second best path is the alternative to the best path. The
            // first path will always be the token itself.
            let nbests = lattice.nbest(2);

            // TODO: Check the correctness of this logic.
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

        let sentences_with_id: Vec<(usize, &str)> = self
            .sentences
            .iter()
            .map(|s| s.as_str())
            .enumerate()
            .collect();

        // For a token ID i, inverted[i] stores the list of sentence IDs that
        // contain the token i.
        // This step computes the global frequency of each token and the list of
        // sentences that contain each token.
        let chunk_size = std::cmp::max(self.sentences.len() / current_num_threads(), 1);
        let (token_frequencies, inverted): (Vec<f64>, Vec<Vec<usize>>) = sentences_with_id
            .maybe_par_chunks(chunk_size)
            .map(|chunk| {
                let mut freq: Vec<f64> = vec![0.0; vocab.len()];
                let mut inverted: Vec<Vec<usize>> = vec![Vec::new(); vocab.len()];

                for (i, sentence) in chunk {
                    let mut lattice = Lattice::from(sentence.as_bytes(), bos_id, eos_id);
                    model.populate_nodes(&mut lattice);

                    for node in lattice.viterbi() {
                        let id = node.borrow().id;
                        freq[id] += 1.0;
                        inverted[id].push(*i);
                    }
                }
                (freq, inverted)
            })
            .reduce(
                || (vec![0.0; vocab.len()], vec![Vec::new(); vocab.len()]),
                |(freq, inverted), (lfreq, linverted)| {
                    (
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

        let sum_token_frequencies: f64 = token_frequencies.iter().sum();
        let logsum_token_frequencies = sum_token_frequencies.ln();

        let mut candidates: Vec<(usize, f64)> = vec![];
        let mut pruned_vocab: Vec<ScoredToken> = Vec::with_capacity(self.vocab_size);

        // Compute how likely the LM likelihood is reduced if the token `i` is
        // removed from the vocabulary.
        // Since the exact computation of loss is difficult, we compute the loss
        // approximately by assuming that all tokens `i` would be replaced with
        // alternatives[i] if removed.
        for (id, (token, score)) in vocab.iter().enumerate() {
            if token_frequencies[id] == 0.0 && !always_keep[id] {
                // Not found in Viterbi path. Can remove this entry safely.
                continue;
            } else if alternatives[id].is_empty() {
                // No alternatives. Keeps this entry.
                pruned_vocab.push((token.clone(), *score));
            } else {
                // The number of sentences in which the token `i` appears.
                let mut f: f64 = inverted[id].len() as f64;

                if f == 0.0 || !f.is_normal() {
                    continue;
                }

                f /= self.sentences.len() as f64; // normalizes by all sentence frequency.

                let logprob = token_frequencies[id].ln() - logsum_token_frequencies;

                // After removing token `i`, its frequency is re-assigned
                // its alternatives.
                let logsum_alternative = (sum_token_frequencies
                    + token_frequencies[id] * (alternatives.len() - 1) as f64)
                    .ln();

                // The frequencies of alternatives are increased by the
                // frequency of the removed token.
                let mut logprob_alternative = 0.0;
                for n in &alternatives[id] {
                    logprob_alternative +=
                        (token_frequencies[*n] + token_frequencies[id]).ln() - logsum_alternative;
                }

                // loss: the diff of likelihood after removing the sentencepieces[i].
                let loss = f * (logprob - logprob_alternative);

                assert!(!loss.is_nan());

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

        for (token, score) in model.iter() {
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
        let mut a = vec![(vec![], 1.0), (vec![], 2.0)];
        to_log_prob(&mut a);
        let scores = a.iter().map(|(_, score)| *score).collect::<Vec<_>>();
        // ln(1) - ln(3)
        assert_approx_eq!(scores[0], -1.098, 0.01);
        // ln(2) - ln(3)
        assert_approx_eq!(scores[1], -0.405, 0.01);

        let mut a = vec![(vec![], 0.0), (vec![], 3.0)];
        to_log_prob(&mut a);
        let scores = a.iter().map(|(_, score)| *score).collect::<Vec<_>>();
        // ln(0) - ln(3)
        assert_eq!(scores[0], f64::NEG_INFINITY);
    }

    #[test]
    fn test_vocab_generator_is_valid_token() {
        // Strict
        let vg = VocabularyGenerator::new(2, 2, 0.0, vec![], true);

        assert!(vg.is_valid_token("hello"));
        assert!(vg.is_valid_token(" hello world"));
        assert!(!vg.is_valid_token("hello "));
        assert!(!vg.is_valid_token("hello world"));
        assert!(!vg.is_valid_token("hello world "));

        assert!(vg.is_valid_token("//U "));
        assert!(vg.is_valid_token(" - "));

        assert!(vg.is_valid_token(" abc"));
        assert!(vg.is_valid_token(" 123"));

        assert!(vg.is_valid_token("大家哦好"));
        assert!(!vg.is_valid_token("Hello 大家哦好"));
        assert!(!vg.is_valid_token("été"));

        assert!(!vg.is_valid_token("DC name"));

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

        let vg = VocabularyGenerator::new(2, 2, 0.0, vec![], false);

        assert!(vg.is_valid_token("D "));
        assert!(vg.is_valid_token("<div>"));
        assert!(vg.is_valid_token("(D self"));

        assert!(vg.is_valid_token("DC name"));
    }
}
