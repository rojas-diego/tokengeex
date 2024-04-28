use std::sync::RwLock;

use fnv::{FnvHashMap, FnvHashSet};
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use regex::Regex;
use tokengeex::{par_chunk_size, Model, ScoredToken, Task, TokenID};

pub struct ModelVocabularyMerger {
    allow: Regex,
    num_merges: usize,
    step: usize,
    scale_factor: f64,
    max_token_length: usize,
}

impl ModelVocabularyMerger {
    pub fn new(
        allow: Regex,
        num_merges: usize,
        step: usize,
        scale_factor: f64,
        max_token_length: usize,
    ) -> Self {
        Self {
            allow,
            num_merges,
            step,
            scale_factor,
            max_token_length,
        }
    }

    pub fn merge(self, model: &mut Model, samples: &[&str]) {
        // Merges that are known to be disallowed.
        let mut ignore = FnvHashSet::<(TokenID, TokenID)>::default();
        let start_vocab_size = model.vocab_size();

        while model.vocab_size() < start_vocab_size + self.num_merges {
            let chunk_size = par_chunk_size(samples.len(), 4);
            let pair_frequencies = RwLock::new(FnvHashMap::<(TokenID, TokenID), usize>::default());
            let task = Task::new(
                &format!(
                    "BPE Merge {}/{}",
                    model.vocab_size() - start_vocab_size,
                    self.num_merges
                ),
                samples.len(),
                chunk_size,
            );

            task.start();

            samples.par_chunks(chunk_size).for_each(|chunk| {
                let mut ltask = task.local(chunk.len());
                let mut local_pair_frequencies = FnvHashMap::<(TokenID, TokenID), usize>::default();

                for sample in chunk {
                    let ids = model.encode(sample).unwrap();

                    for i in 1..ids.len() {
                        let pair = (ids[i - 1], ids[i]);
                        *local_pair_frequencies.entry(pair).or_insert(0) += 1;
                    }

                    ltask.record(sample.len());
                }

                {
                    let mut pair_frequencies = pair_frequencies.write().unwrap();
                    for (pair, freq) in local_pair_frequencies {
                        *pair_frequencies.entry(pair).or_insert(0) += freq;
                    }
                }

                ltask.finish();
            });

            let mut pairs = pair_frequencies
                .into_inner()
                .unwrap()
                .into_iter()
                .collect::<Vec<((TokenID, TokenID), usize)>>();

            pairs.sort_unstable_by(|a, b| b.1.cmp(&a.1));

            let mut merges = std::cmp::min(
                self.step,
                self.num_merges - (model.vocab_size() - start_vocab_size),
            );
            for ((a, b), freq) in pairs.iter().copied() {
                if merges == 0 {
                    break;
                }

                let pair = (a, b);

                let a = model.vocab()[a as usize].clone();
                let b = model.vocab()[b as usize].clone();

                let mut value = a.value.clone();
                value.extend_from_slice(&b.value);
                let score = (a.score + b.score) * self.scale_factor;
                let token = ScoredToken::new(value, score, false);

                if token.len() > self.max_token_length
                    || !self.allow.is_match(&String::from_utf8_lossy(&token.value))
                {
                    if !ignore.contains(&pair) {
                        log::debug!(
                            "Ignoring merge of a={} b={} freq={} into={:?}",
                            a,
                            b,
                            freq,
                            token
                        );
                        ignore.insert(pair);
                    }
                    continue;
                }

                model.add_tokens([token.clone()]);

                merges -= 1;

                log::info!("Merged a={} b={} freq={} into={:?}", a, b, freq, token);
            }

            if merges == self.step {
                log::warn!(
                    "No more merges possible after {} merges, consider increasing the number of merges",
                    model.vocab_size() - start_vocab_size
                );
                break;
            }
        }
    }
}
