extern crate unicode_categories;
use unicode_categories::UnicodeCategories;

/// Identifies the upcoming word as by being capitalized.
pub const CAPITALIZE_CAPCODE: u8 = b'C';
pub const CAPITALIZE_CAPCODE_CHAR: char = CAPITALIZE_CAPCODE as char;
/// Identifies the upcoming word as being uppercase.
pub const UPPERCASE_CAPCODE: u8 = b'U';
pub const UPPERCASE_CAPCODE_CHAR: char = UPPERCASE_CAPCODE as char;
/// Identifies the first character of the next word as being marked for
/// deletion. In practice, it's only used for spaces.
pub const DELETE_CAPCODE: u8 = b'D';
pub const DELETE_CAPCODE_CHAR: char = DELETE_CAPCODE as char;

/// We consider alphabetical sequences that contain an apostrophe as a single
/// word.
const APOSTROPHE: char = '\'';

/// Encode a string into a capcode sequence. Uppercase characters
/// and words are lowercased and losslessly encoded as capcode. Furthermore, all
/// words are expected to start with a space. If a word does not start with a
/// space, an extra space is added and a capcode is prepended to the word.
///
/// # Examples
///
/// ```no_run
/// use tokengeex::capcode::encode;
///
/// let input = "Hello, World!";
/// let output = encode(input);
/// assert_eq!(output, "DC hello,C world!".to_string());
/// ```
///
/// # Details
///
/// The encoding is lossless, meaning that the original string can be recovered
/// from the encoded sequence.
pub fn encode(input: &str) -> String {
    let mut prev_char = '.';

    // We expect the encoded sequence to be longer than the input sequence.
    let mut buffer = Vec::with_capacity(input.len() * 2);

    let mut it = input.chars();
    while let Some(c) = it.next() {
        if c.is_uppercase() || c.is_lowercase() || c.is_number() {
            let (count, kind) = consume_word(c, &mut it.clone());

            if prev_char == ' ' {
                buffer.pop();
            } else {
                buffer.push(DELETE_CAPCODE);
            }

            if kind == WordKind::Capitalized {
                buffer.push(CAPITALIZE_CAPCODE);
            } else if kind == WordKind::Uppercase {
                buffer.push(UPPERCASE_CAPCODE);
            }

            buffer.push(b' ');

            if kind != WordKind::Number {
                for cc in c.to_lowercase() {
                    push_char_to_buffer(&mut buffer, cc);
                }
            } else {
                push_char_to_buffer(&mut buffer, c);
            }

            prev_char = c;
            for _ in 0..count {
                let cc = it.next().unwrap();

                if kind != WordKind::Number {
                    for ccc in cc.to_lowercase() {
                        push_char_to_buffer(&mut buffer, ccc);
                    }
                } else {
                    push_char_to_buffer(&mut buffer, cc);
                }

                prev_char = cc;
            }

            continue;
        }

        push_char_to_buffer(&mut buffer, c);

        prev_char = c;
    }

    String::from_utf8(buffer).unwrap()
}

/// Decode a capcode encoded sequence.
pub fn decode(input: &str) -> String {
    let mut should_uppercase = false;
    let mut should_capitalize = false;
    let mut should_delete = false;
    let mut buffer = Vec::with_capacity(input.len());

    let mut it = input.chars();
    while let Some(c) = it.next() {
        if c == DELETE_CAPCODE_CHAR {
            should_delete = true;
            continue;
        } else if c == CAPITALIZE_CAPCODE_CHAR {
            should_capitalize = true;
            continue;
        } else if c == UPPERCASE_CAPCODE_CHAR {
            should_uppercase = true;
            continue;
        }

        if should_delete {
            should_delete = false;
            continue;
        }

        if c.is_lowercase() || c.is_uppercase() || c.is_number() {
            let kind = if c.is_number() {
                WordKind::Number
            } else {
                WordKind::Lowercase
            };

            if should_capitalize || should_uppercase {
                for cc in c.to_uppercase() {
                    push_char_to_buffer(&mut buffer, cc);
                }
            } else {
                push_char_to_buffer(&mut buffer, c);
            }

            let word_it = it.clone().enumerate();
            let mut count = 0;
            for (i, cc) in word_it {
                if cc == APOSTROPHE && count == i {
                    continue;
                }

                if cc.is_uppercase()
                    || !(cc.is_number() || cc.is_lowercase() || is_unicode_modifier(cc))
                {
                    break;
                }

                if kind == WordKind::Number && !cc.is_number() {
                    break;
                }

                count = i + 1;
            }

            for _ in 0..count {
                let cc = it.next().unwrap();
                if should_uppercase {
                    for ccc in cc.to_uppercase() {
                        push_char_to_buffer(&mut buffer, ccc);
                    }
                    continue;
                }
                push_char_to_buffer(&mut buffer, cc);
            }

            should_capitalize = false;
            should_uppercase = false;

            continue;
        }

        push_char_to_buffer(&mut buffer, c);
    }

    String::from_utf8(buffer).unwrap()
}

#[derive(PartialEq, Debug)]
enum WordKind {
    Unknown,
    Uppercase,
    Lowercase,
    Capitalized,
    Number,
}

/// Consume a word from an iterator and returns how many characters were
/// consumed. In this capcode implementation, a word boundary is tricky to
/// define but goes something like this:
///
/// "{WORD} "
/// "{Word}Word"
/// "{Wor'd}WORD"
/// "{WORD}Word"
/// "{word}Word"
/// "{word}WORD"
/// "{WORD}大家好"
/// "{word}123"
/// "{123}word"
///
/// Where word is any sequence of ascii alphabetical characters
/// or apostrophes or unicode modifiers.
fn consume_word(c: char, it: &mut std::str::Chars) -> (usize, WordKind) {
    let mut count = 0;
    let mut it = it.enumerate().peekable();

    let mut kind = if c.is_number() {
        WordKind::Number
    } else if c.is_lowercase() {
        WordKind::Lowercase
    } else {
        WordKind::Unknown
    };

    while let Some((i, cc)) = it.next() {
        if cc == APOSTROPHE && count == i {
            continue;
        }

        if kind == WordKind::Number {
            if !cc.is_number() && !is_unicode_modifier(cc) {
                break;
            }
        } else {
            if (!cc.is_lowercase() && !cc.is_uppercase()) && !is_unicode_modifier(cc) {
                break;
            }

            if kind == WordKind::Unknown {
                if cc.is_uppercase() {
                    kind = WordKind::Uppercase;
                } else {
                    kind = WordKind::Capitalized;
                }
            } else if cc.is_uppercase()
                && (kind != WordKind::Uppercase
                    || (it.peek().is_some() && it.peek().unwrap().1.is_lowercase()))
            {
                break;
            }

            if kind == WordKind::Uppercase && cc.is_lowercase() {
                break;
            }

            if kind == WordKind::Capitalized && cc.is_uppercase() {
                break;
            }
        }

        count = i + 1;
    }

    (
        count,
        if kind == WordKind::Unknown {
            if c.is_uppercase() {
                WordKind::Uppercase
            } else {
                WordKind::Lowercase
            }
        } else {
            kind
        },
    )
}

/// Check whether the character is a Unicode modifier
pub fn is_unicode_modifier(c: char) -> bool {
    c.is_mark_nonspacing() || c.is_mark_spacing_combining() || c.is_mark_enclosing()
}

/// Check whether a character is capcode marker.
#[allow(unused)]
pub fn is_marker(c: char) -> bool {
    c == CAPITALIZE_CAPCODE_CHAR || c == UPPERCASE_CAPCODE_CHAR || c == DELETE_CAPCODE_CHAR
}

/// Write a single character to a buffer. Grows the buffer as needed.
fn push_char_to_buffer(b: &mut Vec<u8>, c: char) {
    let mut buf = [0; 4];
    let bytes = c.encode_utf8(&mut buf).as_bytes();
    b.extend_from_slice(bytes);
}

mod tests {
    #[cfg(test)]
    use super::*;

    #[cfg(test)]
    const SAMPLES: &[(&str, &str)] = &[
        ("Hello, World!", "DC hello,C world!"),
        ("DON'T, panic!", "DU don't, panic!"),
        ("O''N'T, panic!", "DU o''DU n't, panic!"),
        ("http", "D http"),
        ("Hello世界", "DC hello世界"),
        ("someMIXEDCase123", "D someDU mixedDC caseD 123"),
        ("camelCase", "D camelDC case"),
        ("CamelCase", "DC camelDC case"),
        ("CASE", "DU case"),
        (" UPPER_CASE ", "U upper_DU case "),
        (" lower_case ", " lower_D case "),
        ("CamelCASEHttp", "DC camelDU caseDC http"),
        ("N̥͡m", "DC n̥͡m"),
        ("Été 2020", "DC été 2020"),
        ("Ⅰ", "D Ⅰ"),
        ("Ⅻ", "D Ⅻ"),
    ];

    #[test]
    fn test_encode() {
        for (input, expected) in SAMPLES {
            assert_eq!(encode(input), *expected);
        }
    }

    #[test]
    fn test_decode() {
        for (expected, input) in SAMPLES {
            assert_eq!(decode(input), *expected);
        }
    }

    #[test]
    fn test_consume_word() {
        assert_eq!(
            consume_word('H', &mut "ello, World!".chars()),
            (4, WordKind::Capitalized)
        );
        assert_eq!(
            consume_word('c', &mut "ase".chars()),
            (3, WordKind::Lowercase)
        );
        assert_eq!(
            consume_word('C', &mut "ASE".chars()),
            (3, WordKind::Uppercase)
        );
        assert_eq!(
            consume_word('D', &mut "ON'T panic!".chars()),
            (4, WordKind::Uppercase)
        );
        assert_eq!(
            consume_word('D', &mut "O''N'T panic!".chars()),
            (1, WordKind::Uppercase)
        );
        assert_eq!(
            consume_word('a', &mut "bcdef".chars()),
            (5, WordKind::Lowercase)
        );
        assert_eq!(
            consume_word('C', &mut "amelCase".chars()),
            (4, WordKind::Capitalized)
        );
        assert_eq!(
            consume_word('U', &mut "PPER_CASE".chars()),
            (4, WordKind::Uppercase)
        );
        assert_eq!(
            consume_word('M', &mut "IXEDCase".chars()),
            (4, WordKind::Uppercase)
        );
        assert_eq!(
            consume_word('M', &mut "ixedCASE".chars()),
            (4, WordKind::Capitalized)
        );
        assert_eq!(
            consume_word('1', &mut "23Identifier".chars()),
            (2, WordKind::Number)
        );
        assert_eq!(
            consume_word('C', &mut "hinese好了".chars()),
            (6, WordKind::Capitalized)
        );
        assert_eq!(
            consume_word('D', &mut " ".chars()),
            (0, WordKind::Uppercase)
        );
        assert_eq!(
            consume_word('1', &mut " 2345".chars()),
            (0, WordKind::Number)
        );
        assert_eq!(
            consume_word('D', &mut "o".chars()),
            (1, WordKind::Capitalized)
        );
        assert_eq!(
            consume_word('D', &mut "o ".chars()),
            (1, WordKind::Capitalized)
        );
        assert_eq!(
            consume_word('d', &mut "'ONT".chars()),
            (0, WordKind::Lowercase)
        );
        assert_eq!(
            consume_word('D', &mut "'ont".chars()),
            (4, WordKind::Capitalized)
        );
        assert_eq!(
            consume_word('H', &mut "e\u{0301}".chars()),
            (2, WordKind::Capitalized)
        );

        let mut s = "N̥͡m".chars();

        assert_eq!(
            consume_word(s.next().unwrap(), &mut s.clone()),
            (3, WordKind::Capitalized)
        );

        assert_eq!(
            consume_word('H', &mut "é".chars()),
            (1, WordKind::Capitalized)
        );
    }

    #[test]
    fn test_unicode_properties() {
        // So... there's such as thing as lowercase numbers in unicode...
        assert_eq!('Ⅰ'.to_lowercase().to_string(), "ⅰ");
    }
}
