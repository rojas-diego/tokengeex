// Code imported and modified from: https://github.com/huggingface/tokenizers
// License: https://github.com/huggingface/tokenizers/blob/4a8105c36671ef46738d6e2799c55198139b87b2/LICENSE

use std::collections::HashMap;
use std::hash::Hash;

#[derive(Default)]
pub(crate) struct TrieBuilder<Label, Data> {
    trie: Trie<Label, Data>,
}

impl<Label: Eq + Hash + Copy, Data: Clone> TrieBuilder<Label, Data> {
    pub fn push(&mut self, element: &[Label], data: Data) {
        self.trie.push(element, data);
    }

    pub fn build(self) -> Trie<Label, Data> {
        self.trie
    }
}

#[derive(Clone)]
pub(crate) struct Trie<Label, Data> {
    root: Node<Label, Data>,
}

impl<Label: Eq + Hash + Copy, Data: Clone> Trie<Label, Data> {
    pub fn push(&mut self, element: &[Label], data: Data) {
        let mut node = &mut self.root;
        for label in element.iter() {
            node = node.children.entry(*label).or_default();
        }
        node.data = Some(data);
    }

    pub fn common_prefix_search<T>(&self, iterator: T) -> TrieIterator<Label, Data, T>
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

pub(crate) struct TrieIterator<'a, Label, Data, T> {
    node: &'a Node<Label, Data>,
    prefix: Vec<Label>,
    iterator: T,
}

impl<Label, Data, T> Iterator for TrieIterator<'_, Label, Data, T>
where
    Label: Eq + Hash + Copy,
    Data: Clone,
    T: Iterator<Item = Label>,
{
    type Item = Data;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let label = self.iterator.next()?;
            self.prefix.push(label);
            let child = self.node.children.get(&label)?;
            self.node = child;

            if let Some(data) = &self.node.data {
                return Some(data.clone());
            }
        }
    }
}

impl<Label, Data> Default for Trie<Label, Data> {
    fn default() -> Self {
        Self {
            root: Node::default(),
        }
    }
}

#[derive(Clone)]
struct Node<Label, Data> {
    data: Option<Data>,
    children: HashMap<Label, Node<Label, Data>>,
}

impl<Label, Data> Default for Node<Label, Data> {
    fn default() -> Self {
        Self {
            data: None,
            children: HashMap::new(),
        }
    }
}
