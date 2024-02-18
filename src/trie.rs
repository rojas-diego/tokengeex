// Code imported and modified from: https://github.com/huggingface/tokenizers
// License: https://github.com/huggingface/tokenizers/blob/4a8105c36671ef46738d6e2799c55198139b87b2/LICENSE

use std::collections::HashMap;
use std::hash::Hash;

#[derive(Default)]
pub(crate) struct TrieBuilder<Label> {
    trie: Trie<Label>,
}

impl<Label: Eq + Hash + Copy> TrieBuilder<Label> {
    pub fn push(&mut self, element: &[Label]) {
        self.trie.push(element);
    }

    pub fn build(self) -> Trie<Label> {
        self.trie
    }
}

#[derive(Clone)]
pub(crate) struct Trie<Label> {
    root: Node<Label>,
}

impl<Label: Eq + Hash + Copy> Trie<Label> {
    pub fn push(&mut self, element: &[Label]) {
        let mut node = &mut self.root;
        for label in element.iter() {
            node = node.children.entry(*label).or_default();
        }
        node.is_leaf = true;
    }

    pub fn common_prefix_search<T>(&self, iterator: T) -> TrieIterator<Label, T>
    where
        T: Iterator<Item = Label>,
    {
        TrieIterator {
            node: &self.root,
            prefix: vec![],
            iterator,
        }
    }
}

pub(crate) struct TrieIterator<'a, Label, T> {
    node: &'a Node<Label>,
    prefix: Vec<Label>,
    iterator: T,
}

impl<Label, T> Iterator for TrieIterator<'_, Label, T>
where
    Label: Eq + Hash + Copy,
    T: Iterator<Item = Label>,
{
    type Item = Vec<Label>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let label = self.iterator.next()?;
            self.prefix.push(label);
            let child = self.node.children.get(&label)?;
            self.node = child;
            if self.node.is_leaf {
                return Some(self.prefix.clone());
            }
        }
    }
}

impl<Label> Default for Trie<Label> {
    fn default() -> Self {
        Self {
            root: Node::default(),
        }
    }
}

#[derive(Clone)]
struct Node<Label> {
    is_leaf: bool,
    children: HashMap<Label, Node<Label>>,
}

impl<Label> Default for Node<Label> {
    fn default() -> Self {
        Self {
            is_leaf: false,
            children: HashMap::new(),
        }
    }
}
