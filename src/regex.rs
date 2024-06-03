use regex::Regex;

pub const ANY_CHAR: &str = r#"."#;
pub const LOWERCASE_WORD: &str = r#" ?[a-z]+"#;
pub const UPPERCASE_WORD: &str = r#" ?[A-Z]+"#;
pub const CAPITALIZED_WORD: &str = r#" ?[A-Z][a-z]+"#;
pub const WORD: &str = r#" ?[A-Za-z]+"#;
pub const ENGLISH_WORD: &str = r#" ?[A-Za-z]+'[a-zA-Z]{1,2}"#;
pub const FRENCH_WORD: &str = r#" ?[A-Za-zÀ-ÿ]+"#;
pub const CHINESE_WORD: &str = r#"[\u3400-\u4DBF\u4E00-\u9FFF]+"#;
pub const ENGLISH_CONTRACTION: &str = r#"'(?:re|ve|s|d|ll|t|m)"#;
pub const SPACE_DIGIT: &str = r#" [0-9]"#;
pub const SHORT_NUMBER: &str = r#"[0-9]{1,3}"#;
pub const SPACE_SHORT_NUMBER: &str = r#" [0-9]{1,3}"#;
pub const SHORT_DECIMAL_NUMBER: &str = r#"[0-9]{1,3}\.[0-9]"#;
pub const SPACE_SHORT_DECIMAL_NUMBER: &str = r#" [0-9]{1,3}\.[0-9]"#;
pub const WORD_WRAPPED_IN_BRACKETS: &str = r#"\[[A-Za-z]+\]"#;
pub const SHORT_NUMBER_WRAPPED_IN_BRACKETS: &str = r#"\[[0-9]{1,3}\]"#;
pub const WORD_WRAPPED_IN_QUOTES: &str = r#"['"][A-Za-z]+['"]"#;
pub const WORD_WRAPPED_IN_ANGLE_BRACKETS: &str = r#"<[A-Za-z]+>"#;
pub const PUNCT_WORD: &str = r#"[[:punct:]][A-Za-z]+"#;
pub const SPACE_PUNCT_WORD: &str = r#" [[:punct:]][A-Za-z]+"#;
pub const WORD_PUNCT: &str = r#"[A-Za-z][[:punct:]]"#;
pub const DOT_SHORT_NUMBER: &str = r#"\.[0-9]{1,3}"#;
pub const BRACKET_SHORT_NUMBER: &str = r#"\[[0-9]{1,3}"#;
pub const INDENT: &str = r#"(?:[ ]+)|[\t]+"#;
pub const NEWLINE_INDENT: &str = r#"(?:\n[ ]+)|(?:\n[\t]+)"#;
pub const WHITESPACE: &str = r#"\s+"#;
pub const REPEATED_PUNCT: &str = r#"[[:punct:]]+"#;
pub const FEW_REPEATED_PUNCT: &str = r#"[[:punct:]]{1,4}"#;
pub const REPEATED_PUNCT_SPACE: &str = r#"(?: |[[:punct:]])+"#;
pub const FEW_REPEATED_PUNCT_SPACE: &str = r#"(?: |[[:punct:]]){1,4}"#;
pub const PUNCT_NEWLINE: &str = r#"[[:punct:]]+\n"#;
pub const REPEATED_PUNCT_NEWLINE_INDENT: &str = r#"[[:punct:]]+\n[ \t]+"#;

macro_rules! constexpr_regex {
    ($regex:expr) => {{
        fn generated_function() -> Regex {
            Regex::new($regex).unwrap()
        }
        generated_function as fn() -> Regex
    }};
}

macro_rules! repeated_char_regex {
    ($chars:expr, $min:expr, $max:expr) => {{
        fn generated_function() -> Regex {
            let mut components = Vec::new();

            for c in $chars.chars() {
                let mut regex = String::new();
                regex.push_str(&regex::escape(&c.to_string()));
                regex.push_str("{");
                regex.push_str($min.to_string().as_str());
                regex.push_str(",");
                regex.push_str($max.to_string().as_str());
                regex.push_str("}");
                components.push(regex);
            }

            Regex::new(components.join("|").as_str()).unwrap()
        }
        generated_function as fn() -> Regex
    }};
}

pub const PACKAGE_KEYWORDS: &[&str] = &["package", "import", "export", "module", "use"];

pub const CONTROL_FLOW_STATEMENTS: &[&str] = &[
    "if", "else", "for", "while", "do", "break", "continue", "return", "switch", "case", "default",
    "goto", "try", "catch", "finally", "throw", "assert", "yield", "defer", "await",
];

pub const LITERALS: &[&str] = &[
    "true",
    "false",
    "True",
    "False",
    "null",
    "nil",
    "None",
    "undefined",
];

pub const QUALIFIERS: &[&str] = &[
    "const",
    "static",
    "final",
    "volatile",
    "extern",
    "register",
    "pub",
    "private",
    "protected",
    "public",
    "abstract",
    "virtual",
    "override",
    "inline",
    "constexpr",
    "explicit",
    "implicit",
    "async",
    "signed",
    "unsigned",
];

pub const PRIMITIVE_TYPES: &[&str] = &[
    "void",
    "bool",
    "char",
    "int",
    "short",
    "long",
    "float",
    "double",
    "u8",
    "u16",
    "u32",
    "u64",
    "u128",
    "i8",
    "i16",
    "i32",
    "i64",
    "i128",
    "f32",
    "f64",
    "usize",
    "isize",
    "str",
    "string",
    "byte",
    "rune",
    "uint",
    "int8",
    "int16",
    "int32",
    "int64",
    "int128",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "uint128",
    "float32",
    "float64",
    "uintptr",
    "complex64",
    "complex128",
];

type RegexFnPtr = fn() -> Regex;

pub enum PatternKind {}

pub const PATTERNS: &[(&str, RegexFnPtr, &[&str], &[&str])] = &[
    // Char
    (
        "any-char",
        constexpr_regex!(ANY_CHAR),
        &["好", "A"],
        &["123"],
    ),
    // Words
    (
        "lowercase-word",
        constexpr_regex!(LOWERCASE_WORD),
        &["hello", " world"],
        &["Hello", "HELLO"],
    ),
    (
        "uppercase-word",
        constexpr_regex!(UPPERCASE_WORD),
        &["HELLO", " WORLD"],
        &["Hello", " WoRLD"],
    ),
    (
        "capitalized-word",
        constexpr_regex!(CAPITALIZED_WORD),
        &[" Hello", "Hello"],
        &["HeLlO"],
    ),
    (
        "word",
        constexpr_regex!(WORD),
        &["hello", " Hello", " HeLlO"],
        &["123"],
    ),
    (
        "english-word",
        constexpr_regex!(ENGLISH_WORD),
        &["don't", " You'll", " He's"],
        &["ABC'DEF"],
    ),
    (
        "french-word",
        constexpr_regex!(FRENCH_WORD),
        &["Été", " compliqué"],
        &["مرحبا"],
    ),
    (
        "chinese-word",
        constexpr_regex!(CHINESE_WORD),
        &["你好", "大家好"],
        &["مرحبا"],
    ),
    // Grammar
    (
        "english-contraction",
        constexpr_regex!(ENGLISH_CONTRACTION),
        &["'re", "'ve", "'s", "'d", "'ll", "'t", "'m"],
        &[],
    ),
    // Numbers
    (
        "space-digit",
        constexpr_regex!(SPACE_DIGIT),
        &[" 1", " 2", " 3"],
        &[" 10"],
    ),
    (
        "short-number",
        constexpr_regex!(SHORT_NUMBER),
        &["1", "123", "789"],
        &["1000"],
    ),
    (
        "space-short-number",
        constexpr_regex!(SPACE_SHORT_NUMBER),
        &[" 1", " 123", " 789"],
        &[],
    ),
    (
        "short-decimal-number",
        constexpr_regex!(SHORT_DECIMAL_NUMBER),
        &["1.1", "123.4", "789.9"],
        &["123.456", "1000.0"],
    ),
    (
        "space-short-decimal-number",
        constexpr_regex!(SPACE_SHORT_DECIMAL_NUMBER),
        &[" 1.1", " 123.4", " 789.9"],
        &[" 123.456", " 1000.0"],
    ),
    // Wrapped
    (
        "word-wrapped-in-brackets",
        constexpr_regex!(WORD_WRAPPED_IN_BRACKETS),
        &["[abc]", "[VALUE]"],
        &[],
    ),
    (
        "short-number-wrapped-in-brackets",
        constexpr_regex!(SHORT_NUMBER_WRAPPED_IN_BRACKETS),
        &["[1]", "[123]", "[789]"],
        &[],
    ),
    (
        "word-wrapped-in-quotes",
        constexpr_regex!(WORD_WRAPPED_IN_QUOTES),
        &["'abc'", "\"VALUE\""],
        &[],
    ),
    (
        "word-wrapped-in-angle-brackets",
        constexpr_regex!(WORD_WRAPPED_IN_ANGLE_BRACKETS),
        &["<abc>", "<VALUE>"],
        &[],
    ),
    // Punctuation Word
    (
        "punct-word",
        constexpr_regex!(PUNCT_WORD),
        &["&abc", ":Abc", "+ABC"],
        &[],
    ),
    // Space Punct Word
    (
        "space-punct-word",
        constexpr_regex!(SPACE_PUNCT_WORD),
        &[" &abc", " :Abc", " +ABC"],
        &[],
    ),
    // Word Punctuation
    (
        "word-punct",
        constexpr_regex!(WORD_PUNCT),
        &["a&", "B:", "C+"],
        &[],
    ),
    // Punctuation Number
    (
        "dot-short-number",
        constexpr_regex!(DOT_SHORT_NUMBER),
        &[".1", ".123", ".789"],
        &[".1000"],
    ),
    (
        "bracket-short-number",
        constexpr_regex!(BRACKET_SHORT_NUMBER),
        &["[1", "[123", "[789"],
        &["[1000"],
    ),
    // Whitespace
    (
        "indent",
        constexpr_regex!(INDENT),
        &[" ", "  ", "    ", "\t", "\t\t", "\t\t\t"],
        &["\t "],
    ),
    (
        "newline-indent",
        constexpr_regex!(NEWLINE_INDENT),
        &["\n ", "\n  ", "\n    ", "\n\t\t", "\n\t\t", "\n\t\t\t"],
        &["\n\t "],
    ),
    (
        "whitespace",
        constexpr_regex!(WHITESPACE),
        &[" ", "  ", "    ", "\n", "\n\n", "\t\t", " \n\t"],
        &[],
    ),
    // Punctuation
    (
        "repeated-same-punct",
        repeated_char_regex!("!?#$%^&*()`[]{}<>|/\\+-=", 2, 4),
        &["##", "%%%", "&&&", "(((", "[[["],
        &["#/", "%%%%%", "&%("],
    ),
    (
        "repeated-punct",
        constexpr_regex!(REPEATED_PUNCT),
        &["####", "()[]{}"],
        &["\n#\n#\n#"],
    ),
    (
        "few-repeated-punct",
        constexpr_regex!(FEW_REPEATED_PUNCT),
        &["#", "##", "###", "()", "[]", "{}"],
        &["#####", "()[]{}"],
    ),
    (
        "repeated-punct-space",
        constexpr_regex!(REPEATED_PUNCT_SPACE),
        &[" # ", " ( ", " ) ", " { ", " } ", " != ", ", "],
        &[],
    ),
    (
        "few-repeated-punct-space",
        constexpr_regex!(FEW_REPEATED_PUNCT_SPACE),
        &[" # ", " ( ", " ) ", " { ", " } ", " != ", ", "],
        &[],
    ),
    (
        "punct-newline",
        constexpr_regex!(PUNCT_NEWLINE),
        &[";\n", "]\n", "}\n"],
        &[";\n\n", "]\n\n", "}\n\n"],
    ),
    (
        "repeated-punct-newline-indent",
        constexpr_regex!(REPEATED_PUNCT_NEWLINE_INDENT),
        &[");\n\t\t", "]\n    "],
        &[],
    ),
];

pub fn build_allow_regex<I>(regexes: I) -> Regex
where
    I: IntoIterator<Item = Regex>,
{
    Regex::new(
        &regexes
            .into_iter()
            .map(|r| format!("^(?:{})$", r.as_str()))
            .collect::<Vec<String>>()
            .join("|"),
    )
    .unwrap()
}

pub fn build_mine_regex<I>(regexes: I) -> Regex
where
    I: IntoIterator<Item = Regex>,
{
    Regex::new(
        &regexes
            .into_iter()
            .map(|r| format!("(?:{})", r.as_str()))
            .collect::<Vec<String>>()
            .join("|"),
    )
    .unwrap()
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use regex::Regex;

    use super::*;

    #[test]
    fn test_regexes() {
        for (name, regex, examples, counter_examples) in PATTERNS {
            let re = Regex::new(&format!("^(?:{})$", &regex())).unwrap();
            for &sample in examples.iter() {
                assert!(
                    re.is_match(sample),
                    "Rule {:?} expected to match {:?} ({})",
                    name,
                    sample,
                    &re,
                );
            }
            for &sample in counter_examples.iter() {
                assert!(
                    !re.is_match(sample),
                    "Rule {:?} expected not to match {:?} ({})",
                    name,
                    sample,
                    &re,
                );
            }
        }

        // Ensure there are no duplicate regex names
        let mut names = HashSet::new();
        let mut regexes = HashSet::new();
        for (name, regex, _, _) in PATTERNS {
            assert!(names.insert(name), "Duplicate regex name found: {:?}", name);
            assert!(regexes.insert(regex), "Duplicate regex found: {:?}", regex);
        }
    }
}
