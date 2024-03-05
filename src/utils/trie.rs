// Code imported and modified from: https://github.com/huggingface/tokenizers
// License: https://github.com/huggingface/tokenizers/blob/4a8105c36671ef46738d6e2799c55198139b87b2/LICENSE

#[derive(Default)]
pub(crate) struct TrieBuilder<Data> {
    trie: Trie<Data>,
}

impl<Data: Clone> TrieBuilder<Data> {
    pub fn push(&mut self, element: &[u8], data: Data) {
        self.trie.push(element, data);
    }

    pub fn build(self) -> Trie<Data> {
        self.trie
    }
}

#[derive(Clone)]
pub(crate) struct Trie<Data> {
    root: Node<Data>,
}

impl<Data: Clone> Trie<Data> {
    pub fn push(&mut self, element: &[u8], data: Data) {
        let mut node = &mut self.root;
        for b in element.iter() {
            node = node.children[*b as usize].get_or_insert_with(Default::default);
        }
        node.data = Some(data);
    }

    pub fn common_prefix_search<T>(&self, iterator: T) -> TrieIterator<Data, T>
    where
        T: Iterator<Item = u8>,
    {
        TrieIterator {
            node: &self.root,
            prefix: vec![],
            iterator,
        }
    }
}

pub(crate) struct TrieIterator<'a, Data, T> {
    node: &'a Node<Data>,
    prefix: Vec<u8>,
    iterator: T,
}

impl<Data, T> Iterator for TrieIterator<'_, Data, T>
where
    Data: Clone,
    T: Iterator<Item = u8>,
{
    type Item = Data;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let label = self.iterator.next()?;
            self.prefix.push(label);
            let child = match self.node.children[label as usize].as_ref() {
                Some(child) => child,
                None => return None,
            };
            self.node = child;

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
    children: Vec<Option<Node<Data>>>,
}

impl<Data> Default for Node<Data> {
    fn default() -> Self {
        let mut children = Vec::with_capacity(256);

        for _ in 0..256 {
            children.push(None);
        }

        Self {
            data: None,
            children,
        }
    }
}
