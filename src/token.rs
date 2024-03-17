use std::ops::Deref;

#[derive(Clone, Copy)]
pub struct Token {
    data: [u8; 39],
    len: u8,
}

impl Token {
    pub const MAX_LEN: usize = 39;

    // Returns the number of bytes stored in the Token.
    pub fn len(&self) -> usize {
        self.len as usize
    }

    // Checks if the Token is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // Returns a slice of the data as u8.
    pub fn as_slice(&self) -> &[u8] {
        &self.data[..self.len()]
    }

    // Returns an iterator over the bytes of the Token.
    pub fn iter(&self) -> std::slice::Iter<'_, u8> {
        self.as_slice().iter()
    }
}

impl PartialEq for Token {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for Token {}

impl std::fmt::Debug for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.as_slice();
        let lossy = std::str::from_utf8(data).unwrap_or("<invalid utf-8>");

        f.write_str(lossy)
    }
}

impl From<u8> for Token {
    fn from(byte: u8) -> Self {
        let mut token = Token {
            data: [0; 39],
            len: 1,
        };

        token.data[0] = byte;
        token
    }
}

impl From<&[u8]> for Token {
    fn from(bytes: &[u8]) -> Self {
        if bytes.len() > 39 {
            panic!("String too long for Token conversion");
        }

        let mut token = Token {
            data: [0; 39],
            len: bytes.len() as u8,
        };

        token.data[..bytes.len()].copy_from_slice(bytes);
        token
    }
}

impl From<&str> for Token {
    fn from(s: &str) -> Self {
        Token::from(s.as_bytes())
    }
}

impl From<&&str> for Token {
    fn from(s: &&str) -> Self {
        Token::from(*s)
    }
}

impl From<&Vec<u8>> for Token {
    fn from(v: &Vec<u8>) -> Self {
        Token::from(v.as_slice())
    }
}

impl From<&String> for Token {
    fn from(s: &String) -> Self {
        Token::from(s.as_bytes())
    }
}

impl Deref for Token {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl std::hash::Hash for Token {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use super::Token;

    #[test]
    fn test_len_and_is_empty_on_empty_token() {
        let token = Token::from("");
        assert_eq!(token.len(), 0);
        assert!(token.is_empty());
    }

    #[test]
    fn test_len_on_non_empty_token() {
        let token = Token::from("hello");
        assert_eq!(token.len(), 5);
        assert!(!token.is_empty());
    }

    #[test]
    fn test_from_str_with_various_lengths() {
        for i in 1..Token::MAX_LEN {
            let s: String = "a".repeat(i);
            let token = Token::from(s.as_str());
            assert_eq!(token.len(), i, "Failed at length {}", i);
        }
    }

    #[test]
    fn test_as_slice() {
        let s = "hello";
        let token = Token::from(s);
        assert_eq!(token.as_slice(), s.as_bytes());
    }

    #[test]
    fn test_iter() {
        let s = "hello";
        let token = Token::from(s);
        let mut iter = token.iter();
        assert_eq!(iter.next(), Some(&b'h'));
        assert_eq!(iter.next(), Some(&b'e'));
        assert_eq!(iter.next(), Some(&b'l'));
        assert_eq!(iter.next(), Some(&b'l'));
        assert_eq!(iter.next(), Some(&b'o'));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_debug() {
        let s = "hello";
        let token = Token::from(s);
        assert_eq!(format!("{:?}", token), s);
    }

    #[test]
    fn hash_equality_for_identical_tokens() {
        let token_a = Token::from("test data");
        let token_b = Token::from("test data");

        let mut hasher_a = DefaultHasher::new();
        token_a.hash(&mut hasher_a);
        let hash_a = hasher_a.finish();

        let mut hasher_b = DefaultHasher::new();
        token_b.hash(&mut hasher_b);
        let hash_b = hasher_b.finish();

        assert_eq!(hash_a, hash_b, "Hashes should be equal for identical data");
    }

    #[test]
    fn hash_inequality_for_different_tokens() {
        let token_a = Token::from("test data");
        let token_b = Token::from("different data");

        let mut hasher_a = DefaultHasher::new();
        token_a.hash(&mut hasher_a);
        let hash_a = hasher_a.finish();

        let mut hasher_b = DefaultHasher::new();
        token_b.hash(&mut hasher_b);
        let hash_b = hasher_b.finish();

        assert_ne!(
            hash_a, hash_b,
            "Hashes should be different for different data"
        );
    }
}
