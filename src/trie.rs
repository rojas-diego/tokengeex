// Code imported and modified from: https://github.com/huggingface/tokenizers
// License: https://github.com/huggingface/tokenizers/blob/4a8105c36671ef46738d6e2799c55198139b87b2/LICENSE

use std::collections::HashMap;

#[derive(Clone)]
pub struct Trie<Data> {
    root: Node<Data>,
}

impl<Data: Clone> Trie<Data> {
    pub fn push(&mut self, element: &[u8], data: Data) {
        let mut node = &mut self.root;

        for b in element.iter() {
            node = node.children.entry(*b).or_default();
        }

        node.data = Some(data);
    }

    pub fn common_prefix_search<'a, T>(
        &self,
        iterator: T,
        prefix: &'a mut Vec<u8>,
    ) -> TrieIterator<'_, 'a, Data, T>
    where
        T: Iterator<Item = u8>,
    {
        TrieIterator {
            node: &self.root,
            prefix,
            iterator,
        }
    }
}

pub struct TrieIterator<'a, 'b, Data, T> {
    node: &'a Node<Data>,
    prefix: &'b mut Vec<u8>,
    iterator: T,
}

impl<Data, T> Iterator for TrieIterator<'_, '_, Data, T>
where
    Data: Clone,
    T: Iterator<Item = u8>,
{
    type Item = Data;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let b = self.iterator.next()?;
            self.prefix.push(b);
            self.node = match self.node.children.get(&b) {
                Some(child) => child,
                None => return None,
            };
            if let Some(data) = &self.node.data {
                return Some(data.clone());
            }
        }
    }
}

impl<Data> Default for Trie<Data> {
    fn default() -> Self {
        Self {
            root: Node::default(),
        }
    }
}

#[derive(Clone)]
struct Node<Data> {
    data: Option<Data>,
    children: HashMap<u8, Node<Data>, fnv::FnvBuildHasher>,
}

impl<Data> Default for Node<Data> {
    fn default() -> Self {
        Self {
            data: None,
            children: HashMap::default(),
        }
    }
}
