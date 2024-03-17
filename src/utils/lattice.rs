// Code imported and modified from: https://github.com/huggingface/tokenizers
// License: https://github.com/huggingface/tokenizers/blob/4a8105c36671ef46738d6e2799c55198139b87b2/LICENSE

use std::cell::RefCell;
use std::cmp::{min, Ordering};
use std::collections::BinaryHeap;
use std::rc::Rc;

use crate::TokenID;

/// Structure to implement Viterbi algorithm to find the best encoding, or
/// sample from all possible encodings of a given sentence.
#[derive(Debug)]
pub struct Lattice<'a> {
    /// The sentence to be tokenized.
    pub sentence: &'a [u8],
    /// An array which keeps track of all the tokens which begin at a given
    /// position in the sentence.
    pub begin_nodes: Vec<Vec<usize>>,
    /// An array which keeps track of all the tokens which end at a given
    /// position in the sentence.
    pub end_nodes: Vec<Vec<usize>>,
    /// Each node represents a token in the sentence.
    pub nodes: Vec<Node>,

    bos_idx: usize,
    eos_idx: usize,
}

/// A node from the lattice, that helps reconstruct the underlying `String`
#[derive(Debug, Clone)]
pub struct Node {
    /// Position in the sentence.
    pub pos: usize,
    /// Token ID.
    pub token_id: TokenID,
    /// Length of the token.
    pub token_len: usize,
    /// Score of the token.
    pub score: f64,
    /// Previous node.
    pub prev: Option<usize>,
    /// Score obtained from backtracking.
    pub backtrack_score: f64,
}

impl PartialEq for Node {
    fn eq(&self, other: &Node) -> bool {
        self.token_id == other.token_id
    }
}

impl Node {
    pub fn new(pos: usize, token_id: TokenID, token_len: usize, score: f64) -> Self {
        Self {
            pos,
            token_id,
            token_len,
            score,
            prev: None,
            backtrack_score: 0.0,
        }
    }
}

impl<'a> Lattice<'a> {
    pub fn new() -> Self {
        Self {
            sentence: &[],
            nodes: Vec::with_capacity(1024 * 1024),
            begin_nodes: vec![],
            end_nodes: vec![],
            bos_idx: 0,
            eos_idx: 0,
        }
    }

    pub fn from(
        &mut self,
        sentence: &'a [u8],
        bos_id: TokenID,
        eos_id: TokenID,
        vec_pool: &mut VecPool,
    ) {
        self.sentence = sentence;
        self.nodes.clear();

        // Return existing Vecs to the pool and borrow new ones as needed.
        for vec in self.begin_nodes.drain(..) {
            vec_pool.put(vec);
        }
        for vec in self.end_nodes.drain(..) {
            vec_pool.put(vec);
        }

        self.begin_nodes
            .resize_with(sentence.len() + 1, || vec_pool.get());
        self.end_nodes
            .resize_with(sentence.len() + 1, || vec_pool.get());

        // Reinitialize with BOS and EOS nodes.
        self.nodes.push(Node::new(0, bos_id, 0, 0.0));
        self.bos_idx = 0;
        self.nodes.push(Node::new(sentence.len(), eos_id, 0, 0.0));
        self.eos_idx = 1;
        self.end_nodes[0].push(self.bos_idx);
        self.begin_nodes[sentence.len()].push(self.eos_idx);
    }

    pub fn insert(&mut self, pos: usize, token_id: TokenID, token_len: usize, score: f64) {
        let node_idx = self.nodes.len();
        self.begin_nodes[pos].push(node_idx);
        self.end_nodes[pos + token_len].push(node_idx);
        self.nodes.push(Node::new(pos, token_id, token_len, score));
    }

    pub fn viterbi(&mut self) -> Vec<Node> {
        let sentence_len = self.sentence.len();

        for pos in 0..=sentence_len {
            for &rnode_idx in &self.begin_nodes[pos] {
                self.nodes[rnode_idx].prev = None;

                let mut best_score = 0.0;
                let mut best_node = None;

                for &lnode_idx in &self.end_nodes[pos] {
                    let score = self.nodes[lnode_idx].backtrack_score + self.nodes[rnode_idx].score;

                    if best_node.is_none() || score > best_score {
                        best_node = Some(lnode_idx);
                        best_score = score;
                    }
                }

                if best_node.is_none() {
                    return vec![];
                }

                self.nodes[rnode_idx].prev = best_node;
                self.nodes[rnode_idx].backtrack_score = best_score;
            }
        }

        let mut results = Vec::with_capacity(sentence_len / 4);
        let mut node_idx = self.begin_nodes[sentence_len][0];
        while let Some(prev_node_idx) = self.nodes[node_idx].prev {
            results.push(self.nodes[prev_node_idx].clone());
            node_idx = prev_node_idx;
        }

        results.reverse();
        results
    }

    pub fn nbest(&mut self, n: usize) -> Vec<Vec<Node>> {
        match n {
            0 => vec![],
            1 => vec![self.viterbi()],
            _ => {
                let mut agenda: Agenda = BinaryHeap::new();
                let mut hypotheses: Vec<Vec<usize>> = vec![];

                let eos_id = 1;
                let score = self.nodes[eos_id].score;

                let hypo = Hypothesis::new(eos_id, None, score, score);
                agenda.push(hypo);

                self.viterbi();

                while !agenda.is_empty() {
                    let top = Rc::new(RefCell::new(agenda.pop().unwrap()));
                    let node_idx = top.borrow().node_idx;
                    let node_id = self.nodes[node_idx].token_id;
                    let bos_node_id = self.nodes[self.bos_idx].token_id;
                    let node_pos = self.nodes[node_idx].pos;

                    if node_id == bos_node_id {
                        let mut hypothesis = vec![];
                        let mut next: HypothesisRef =
                            Rc::clone(top.borrow().next.as_ref().unwrap());

                        while next.borrow().next.is_some() {
                            hypothesis.push(next.borrow().node_idx);

                            let c: HypothesisRef = next.clone();

                            next = Rc::clone(c.borrow().next.as_ref().unwrap());
                        }

                        hypotheses.push(hypothesis);

                        if hypotheses.len() == n {
                            return hypotheses
                                .iter()
                                .map(|indices| {
                                    indices.iter().map(|&i| self.nodes[i].clone()).collect()
                                })
                                .collect();
                        }
                    } else {
                        for &lnode in &self.end_nodes[node_pos] {
                            let top_gx = top.borrow().gx;
                            let fx = self.nodes[lnode].backtrack_score + top_gx;
                            let gx = self.nodes[lnode].score + top_gx;
                            let hyp = Hypothesis::new(lnode, Some(Rc::clone(&top)), fx, gx);
                            agenda.push(hyp);
                        }
                        // When the input is too long or contains duplicated phrases,
                        // `agenda` will get extremely big. Here we avoid this case by
                        // dynamically shrinking the agenda.
                        let k_max_agenda_size = 100_000;
                        let k_min_agenda_size = 512;
                        if agenda.len() > k_max_agenda_size {
                            let mut new_agenda = BinaryHeap::new();
                            let len = min(k_min_agenda_size, n * 10);
                            for _i in 0..len {
                                new_agenda.push(agenda.pop().unwrap());
                            }
                            agenda = new_agenda;
                        }
                    }
                }

                hypotheses
                    .iter()
                    .map(|indices| indices.iter().map(|&i| self.nodes[i].clone()).collect())
                    .collect()
            }
        }
    }

    /// Computes the marginal probability for each node (token) which is the
    /// probability of this token being part of the optimal segmentation of the
    /// sentence. Returns the normalisation constant which is the probability
    /// of reaching the end of the sentence from the beginning which in itself
    /// corresponds to the probability of the sentence.
    pub fn populate_marginal(&self, expected: &mut [f64]) -> f64 {
        let len = self.sentence.len();
        let num_nodes = self.nodes.len();

        // Initialize alpha (forward probabilities) and beta (backward
        // probabilities) vectors. They measure the log probabilities of
        // reaching a particular node (token) from the start (alpha) or end
        // (beta) of the lattice.
        // - alpha[i] is the log probability of reaching node i from the bos
        // - beta[i] is the log probability of reaching the eos from node i
        let mut alpha = vec![0.0; num_nodes];
        let mut beta = vec![0.0; num_nodes];

        // Calculate forward probabilities (alpha)
        for pos in 0..=len {
            for &rid in &self.begin_nodes[pos] {
                for &lid in &self.end_nodes[pos] {
                    // Update alpha for the right node with log-sum-exp to
                    // prevent underflow, adding the score from the left node
                    // and its alpha
                    alpha[rid] = log_sum_exp(
                        alpha[rid],
                        self.nodes[lid].score + alpha[lid],
                        lid == self.end_nodes[pos][0],
                    );
                }
            }
        }

        // Calculate backward probabilities (beta)
        for pos in (0..=len).rev() {
            for &lid in &self.end_nodes[pos] {
                for &rid in &self.begin_nodes[pos] {
                    // Update beta for the left node similarly, ensuring total
                    // path probability is accumulated
                    beta[lid] = log_sum_exp(
                        beta[lid],
                        self.nodes[rid].score + beta[rid],
                        rid == self.begin_nodes[pos][0],
                    );
                }
            }
        }

        // Calculate the normalization constant (z) from the EOS node's alpha
        let eos_idx = self.eos_idx;
        let z = alpha[eos_idx];

        // Update the expected frequencies for each node based on its marginal
        // probability
        for pos in 0..len {
            for &node_idx in &self.begin_nodes[pos] {
                let id = self.nodes[node_idx].token_id;
                let score = self.nodes[node_idx].score;
                let a = alpha[node_idx];
                let b = beta[node_idx];

                // Calculate the total path probability through this node,
                // subtract the normalization constant and update expected
                // frequencies.
                let total = a + score + b - z;
                let update = total.exp();
                expected[id as usize] += update;
            }
        }

        z
    }
}

impl<'a> Default for Lattice<'a> {
    fn default() -> Self {
        Self::new()
    }
}

fn log_sum_exp(x: f64, y: f64, init_mode: bool) -> f64 {
    if init_mode {
        y
    } else {
        let (vmin, vmax) = if x > y { (y, x) } else { (x, y) };
        let k_minus_log_epsilon = 50.0;
        if vmax > vmin + k_minus_log_epsilon {
            vmax
        } else {
            vmax + ((vmin - vmax).exp() + 1.0).ln()
        }
    }
}

type HypothesisRef = Rc<RefCell<Hypothesis>>;
type Agenda = BinaryHeap<Hypothesis>;

struct Hypothesis {
    node_idx: usize,
    next: Option<HypothesisRef>,
    fx: f64,
    gx: f64,
}

impl Hypothesis {
    pub fn new(node_idx: usize, next: Option<HypothesisRef>, fx: f64, gx: f64) -> Self {
        Self {
            node_idx,
            next,
            fx,
            gx,
        }
    }
}

impl PartialEq for Hypothesis {
    fn eq(&self, other: &Self) -> bool {
        self.fx == other.fx
    }
}

impl Eq for Hypothesis {}

impl PartialOrd for Hypothesis {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Hypothesis {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.fx < other.fx {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    }
}

#[derive(Default)]
pub struct VecPool {
    pool: Vec<Vec<usize>>,
}

impl VecPool {
    pub fn with_capacity(a: usize, b: usize) -> Self {
        Self {
            pool: vec![Vec::with_capacity(b); a],
        }
    }
    // Get a Vec<usize> from the pool, or create a new one if the pool is empty.
    pub fn get(&mut self) -> Vec<usize> {
        self.pool.pop().unwrap_or_default()
    }

    // Return a Vec<usize> to the pool for future reuse.
    pub fn put(&mut self, mut vec: Vec<usize>) {
        vec.clear(); // Clear the vector before putting it back in the pool.
        self.pool.push(vec);
    }
}
