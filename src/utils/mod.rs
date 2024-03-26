use crate::ScoredToken;

pub mod capcode;
pub mod lattice;
pub mod parallelism;
pub mod task;
pub mod trie;

pub fn logprobs(pieces: &mut [ScoredToken]) {
    let sum: f64 = pieces.iter().map(|(_, score)| score).sum();
    let logsum = sum.ln();
    for (_, score) in pieces.iter_mut() {
        *score = score.ln() - logsum;
    }
}
