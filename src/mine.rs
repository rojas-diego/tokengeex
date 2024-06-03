use std::{cmp::Reverse, collections::HashMap};

use dashmap::DashMap;
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use regex::Regex;
use tokengeex::par_chunk_size;

pub struct IdiomMiner {
    num_idioms: usize,
    pattern: Regex,
}

impl IdiomMiner {
    pub fn new(num_idioms: usize, pattern: Regex) -> IdiomMiner {
        IdiomMiner {
            num_idioms,
            pattern,
        }
    }

    pub fn mine(&self, samples: &[&str]) -> Vec<(String, usize)> {
        let frequencies = DashMap::new();
        let chunk_size = par_chunk_size(samples.len(), 5);

        samples.par_chunks(chunk_size).for_each(|chunk| {
            let thread_local_pattern = self.pattern.clone();
            let mut sample_tokens = HashMap::new();

            for sample in chunk {
                for part in thread_local_pattern.find_iter(sample) {
                    let part = part.as_str();
                    *sample_tokens.entry(part).or_insert(0) += 1;
                }
            }

            for (token, count) in sample_tokens {
                *frequencies.entry(token).or_insert(0) += count;
            }
        });

        let mut frequencies = frequencies.into_iter().collect::<Vec<_>>();
        frequencies.sort_by_key(|(_, count)| Reverse(*count));
        frequencies.truncate(self.num_idioms);
        frequencies
            .into_iter()
            .map(|(token, count)| (token.to_string(), count))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use regex::Regex;

    #[test]
    fn test_mine() {
        let samples = &[
            "std::string",
            "std::vector",
            "std::vector<std::string>",
            "std::map<int, std::string>",
        ];

        let pattern = Regex::new(r"std::\w+").unwrap();
        let miner = IdiomMiner::new(2, pattern);
        let idioms = miner.mine(samples);

        assert_eq!(
            idioms,
            vec![
                ("std::string".to_string(), 3),
                ("std::vector".to_string(), 2)
            ]
        );
    }
}
