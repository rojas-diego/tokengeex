// Code imported and modified from: https://github.com/huggingface/tokenizers
// License: https://github.com/huggingface/tokenizers/blob/4a8105c36671ef46738d6e2799c55198139b87b2/LICENSE

use rand::distributions::WeightedIndex;
use rand::prelude::*;
use std::cell::RefCell;
use std::cmp::{min, Ordering};
use std::collections::BinaryHeap;
use std::rc::Rc;

type NodeRef = Rc<RefCell<Node>>;
type HypothesisRef = Rc<RefCell<Hypothesis>>;
type Agenda = BinaryHeap<Hypothesis>;

struct Hypothesis {
    node_ref: NodeRef,
    next: Option<HypothesisRef>,
    fx: f64,
    gx: f64,
}

impl Hypothesis {
    pub fn new(node_ref: NodeRef, next: Option<HypothesisRef>, fx: f64, gx: f64) -> Self {
        Self {
            node_ref,
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

/// Structure to implement Viterbi algorithm to find the best encoding, or
/// sample from all possible encodings of a given sentence.
#[derive(Debug)]
pub(crate) struct Lattice<'a> {
    /// The sentence to be tokenized.
    pub(crate) sentence: &'a [u8],
    /// An array which keeps track of all the tokens which begin at a given
    /// position in the sentence.
    pub(crate) begin_nodes: Vec<Vec<NodeRef>>,
    /// An array which keeps track of all the tokens which end at a given
    /// position in the sentence.
    pub(crate) end_nodes: Vec<Vec<NodeRef>>,

    nodes: Vec<NodeRef>,
    len: usize,
}

impl std::fmt::Display for Lattice<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let display_pieces = |nodes: &Vec<Vec<NodeRef>>| {
            nodes
                .iter()
                .map(|l| {
                    l.iter()
                        .map(|n| self.piece(&n.borrow()))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };

        f.debug_struct("Lattice")
            .field("sentence", &self.sentence)
            .field("begin_nodes", &display_pieces(&self.begin_nodes))
            .field("end_nodes", &display_pieces(&self.end_nodes))
            .finish()
    }
}

/// A node from the lattice, that helps reconstruct the underlying `String`
#[derive(Debug, Clone)]
pub(crate) struct Node {
    /// Token ID.
    pub(crate) id: usize,
    /// ID of the node in the lattice.
    pub(crate) node_id: usize,
    /// Position in the sentence.
    pub(crate) pos: usize,
    /// Length of the token.
    pub(crate) length: usize,
    /// Previous node.
    pub(crate) prev: Option<NodeRef>,

    backtrace_score: f64,
    score: f64,
}

impl PartialEq for Node {
    fn eq(&self, other: &Node) -> bool {
        self.id == other.id
    }
}

impl Node {
    pub(crate) fn new(id: usize, node_id: usize, pos: usize, length: usize, score: f64) -> Self {
        Self {
            id,
            node_id,
            pos,
            length,
            prev: None,
            score,
            backtrace_score: 0.0,
        }
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

impl<'a> Lattice<'a> {
    #[allow(unused)]
    pub(crate) fn from(sentence: &'a [u8], bos_id: usize, eos_id: usize) -> Self {
        let len = sentence.len();

        let k_reserved_node_size = 16;

        // We are adding 2 tokens, bos and eos
        let mut nodes: Vec<NodeRef> = Vec::with_capacity(k_reserved_node_size);
        let mut begin_nodes = vec![Vec::with_capacity(k_reserved_node_size); len + 1];
        let mut end_nodes = vec![Vec::with_capacity(k_reserved_node_size); len + 1];

        let bos = Rc::new(RefCell::new(Node::new(bos_id, 0, 0, 0, 0.0)));
        let eos = Rc::new(RefCell::new(Node::new(eos_id, 1, len, 0, 0.0)));

        begin_nodes[len].push(Rc::clone(&eos));
        end_nodes[0].push(Rc::clone(&bos));

        nodes.push(bos);
        nodes.push(eos);

        Self {
            sentence,
            len,
            nodes,
            begin_nodes,
            end_nodes,
        }
    }

    pub(crate) fn insert(&mut self, pos: usize, length: usize, score: f64, id: usize) {
        let node_id = self.nodes.len();
        let node = Rc::new(RefCell::new(Node::new(id, node_id, pos, length, score)));

        self.begin_nodes[pos].push(Rc::clone(&node));
        self.end_nodes[pos + length].push(Rc::clone(&node));

        self.nodes.push(node);
    }

    pub(crate) fn viterbi(&mut self) -> Vec<NodeRef> {
        let len = self.len;
        let mut pos = 0;
        while pos <= len {
            if self.begin_nodes[pos].is_empty() {
                return vec![];
            }
            for rnode in &self.begin_nodes[pos] {
                rnode.borrow_mut().prev = None;
                let mut best_score = 0.0;
                let mut best_node: Option<NodeRef> = None;
                for lnode in &self.end_nodes[pos] {
                    let score = lnode.borrow().backtrace_score + rnode.borrow().score;
                    if best_node.is_none() || score > best_score {
                        // TODO can we remove this clone ?
                        best_node = Some(lnode.clone());
                        best_score = score
                    }
                }
                match best_node {
                    Some(bnode) => {
                        rnode.borrow_mut().prev = Some(Rc::clone(&bnode));
                        rnode.borrow_mut().backtrace_score = best_score;
                    }
                    None => return vec![],
                }
            }
            if self.sentence[pos..].iter().next().is_some() {
                pos += 1;
            } else {
                break;
            }
        }

        let mut results: Vec<NodeRef> = vec![];
        let root = self.begin_nodes[len][0].borrow();
        let prev = root.prev.as_ref();
        if prev.is_none() {
            return vec![];
        }
        let mut node: NodeRef = prev.unwrap().clone();
        while node.borrow().prev.is_some() {
            results.push(node.clone());
            let n = node.borrow().clone();
            node = n.prev.as_ref().unwrap().clone();
        }
        results.reverse();
        results
    }

    pub(crate) fn piece(&self, node: &Node) -> String {
        String::from_utf8_lossy(&self.sentence[node.pos..node.pos + node.length]).to_string()
    }

    #[allow(dead_code)]
    pub(crate) fn tokens(&mut self) -> Vec<String> {
        self.viterbi()
            .iter()
            .map(|node| self.piece(&node.borrow()))
            .collect()
    }

    pub(crate) fn nbest(&mut self, n: usize) -> Vec<Vec<NodeRef>> {
        match n {
            0 => vec![],
            1 => vec![self.viterbi()],
            _ => {
                // let k_reserved_hypothesis_size = 512;
                let mut agenda: Agenda = BinaryHeap::new();
                let mut hypotheses: Vec<Vec<NodeRef>> = vec![];
                let eos = self.eos_node();
                let score = eos.borrow().score;
                let hypo = Hypothesis::new(eos, None, score, score);
                agenda.push(hypo);

                // Fill backtrace scores
                self.viterbi();

                while !agenda.is_empty() {
                    let top = Rc::new(RefCell::new(agenda.pop().unwrap()));
                    let node = Rc::clone(&top.borrow().node_ref);
                    if node.borrow().id == self.bos_node().borrow().id {
                        let mut hypothesis = vec![];
                        let mut next: HypothesisRef =
                            Rc::clone(top.borrow().next.as_ref().unwrap());
                        while next.borrow().next.is_some() {
                            hypothesis.push(next.borrow().node_ref.clone());
                            let c: HypothesisRef = next.clone();
                            next = Rc::clone(c.borrow().next.as_ref().unwrap());
                        }
                        hypotheses.push(hypothesis);
                        if hypotheses.len() == n {
                            return hypotheses;
                        }
                    } else {
                        for lnode in &self.end_nodes[node.borrow().pos] {
                            let top_gx = top.borrow().gx;
                            let fx = lnode.borrow().backtrace_score + top_gx;
                            let gx = lnode.borrow().score + top_gx;
                            let hyp =
                                Hypothesis::new(Rc::clone(lnode), Some(Rc::clone(&top)), fx, gx);
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
            }
        }
    }

    #[allow(dead_code)]
    pub(crate) fn nbest_tokens(&mut self, n: usize) -> Vec<Vec<String>> {
        self.nbest(n)
            .iter()
            .map(|v| v.iter().map(|node| self.piece(&node.borrow())).collect())
            .collect()
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn bos_node(&self) -> NodeRef {
        Rc::clone(&self.end_nodes[0][0])
    }

    pub(crate) fn eos_node(&self) -> NodeRef {
        Rc::clone(&self.begin_nodes[self.len][0])
    }

    #[allow(dead_code)]
    pub(crate) fn sentence(&self) -> &[u8] {
        self.sentence
    }

    /// Computes the marginal probability for each node (token) which is the
    /// probability of this token being part of the optimal segmentation of the
    /// sentence. Returns the normalisation constant which is the probability
    /// of reaching the end of the sentence from the beginning which in itself
    /// corresponds to the probability of the sentence.
    pub(crate) fn populate_marginal(&self, expected: &mut [f64]) -> f64 {
        let len = self.len();
        let n_nodes = self.nodes.len();

        // Initialize alpha (forward probabilities) and beta (backward
        // probabilities) vectors. They measure the log probabilities of
        // reaching a particular node (token) from the start (alpha) or end
        // (beta) of the lattice.
        // - alpha[i] is the log probability of reaching node i from the bos
        // - beta[i] is the log probability of reaching the eos from node i
        let mut alpha = vec![0.0; n_nodes];
        let mut beta = vec![0.0; n_nodes];

        // Calculate forward probabilities (alpha)
        for pos in 0..=len {
            for rnode in &self.begin_nodes[pos] {
                for lnode in &self.end_nodes[pos] {
                    let lid = lnode.borrow().node_id;
                    let rid = rnode.borrow().node_id;
                    // Update alpha for the right node with log-sum-exp to
                    // prevent underflow, adding the score from the left node
                    // and its alpha
                    alpha[rid] = log_sum_exp(
                        alpha[rid],
                        lnode.borrow().score + alpha[lid],
                        *lnode == self.end_nodes[pos][0],
                    );
                }
            }
        }

        // Calculate backward probabilities (beta)
        for pos in (0..=len).rev() {
            for lnode in &self.end_nodes[pos] {
                for rnode in &self.begin_nodes[pos] {
                    let lid = lnode.borrow().node_id;
                    let rid = rnode.borrow().node_id;

                    // Update beta for the left node similarly, ensuring total
                    // path probability is accumulated
                    beta[lid] = log_sum_exp(
                        beta[lid],
                        rnode.borrow().score + beta[rid],
                        *rnode == self.begin_nodes[pos][0],
                    );
                }
            }
        }

        // Calculate the normalization constant (z) from the EOS node's alpha
        let eos_id = self.begin_nodes[len][0].borrow().node_id;
        let z = alpha[eos_id];

        // Update the expected frequencies for each node based on its marginal
        // probability
        for pos in 0..len {
            for node in &self.begin_nodes[pos] {
                let node_id = node.borrow().node_id;
                let id = node.borrow().id;
                let a = alpha[node_id];
                let b = beta[node_id];

                // Calculate the total path probability through this node,
                // subtract the normalization constant and update expected
                // frequencies.
                let total = a + node.borrow().score + b - z;
                let update = total.exp();
                expected[id] += update;
            }
        }

        z
    }

    #[allow(dead_code)]
    pub(crate) fn sample(&self, theta: f64) -> Vec<NodeRef> {
        let len = self.len();
        if len == 0 {
            return vec![];
        }
        let mut alpha = vec![0.0; self.nodes.len()];
        for pos in 0..=len {
            for rnode in &self.begin_nodes[pos] {
                for lnode in &self.end_nodes[pos] {
                    let lid = lnode.borrow().node_id;
                    let rid = rnode.borrow().node_id;
                    alpha[rid] = log_sum_exp(
                        alpha[rid],
                        theta * (lnode.borrow().score + alpha[lid]),
                        *lnode == self.end_nodes[pos][0],
                    );
                }
            }
        }

        let mut rng = thread_rng();
        let mut results: Vec<NodeRef> = vec![];
        let mut probs: Vec<f64> = vec![];
        let mut z = alpha[self.eos_node().borrow().node_id];
        let mut node = self.eos_node();
        loop {
            probs.clear();
            let pos = node.borrow().pos;
            for lnode in &self.end_nodes[pos] {
                let lid = lnode.borrow().node_id;
                probs.push((alpha[lid] + theta * lnode.borrow().score - z).exp())
            }
            let dist = WeightedIndex::new(&probs).unwrap();
            let index = dist.sample(&mut rng);
            node = Rc::clone(&self.end_nodes[pos][index]);
            if node == self.bos_node() {
                break;
            }
            z = alpha[node.borrow().node_id];
            results.push(Rc::clone(&node));
        }
        results.reverse();
        results
    }

    #[allow(dead_code)]
    pub(crate) fn sample_token(&self, theta: f64) -> Vec<String> {
        self.sample(theta)
            .iter()
            .map(|node| self.piece(&node.borrow()))
            .collect()
    }
}
