use regex::Regex;

pub const ANY_CHAR: &str = r#"."#;

// Word.
const LOWERCASE_WORD: &str = r#"[a-z]+"#;
const UPPERCASE_WORD: &str = r#"[A-Z]+"#;
const CAPITALIZED_WORD: &str = r#"[A-Z][a-z]+"#;
const WORD: &str = r#"[A-Za-z]+"#;
const CHINESE_WORD: &str = r#"[\u3400-\u4DBF\u4E00-\u9FFF]+"#;
// Space word.
const SPACE_LOWERCASE_WORD: &str = r#" ?[a-z]+"#;
const SPACE_UPPERCASE_WORD: &str = r#" ?[A-Z]+"#;
const SPACE_CAPITALIZED_WORD: &str = r#" ?[A-Z][a-z]+"#;
const SPACE_WORD: &str = r#" ?[A-Za-z]+"#;
const SPACE_ENGLISH_WORD: &str = r#" ?[A-Za-z]+'[a-zA-Z]{1,2}"#;
const SPACE_FRENCH_WORD: &str = r#" ?[A-Za-zÀ-ÿ]+"#;
// Grammar.
const ENGLISH_CONTRACTION: &str = r#"'(?:re|ve|s|d|ll|t|m)"#;
// Numbers.
const SPACE_DIGIT: &str = r#" [0-9]"#;
const SHORT_NUMBER: &str = r#"[0-9]{1,3}"#;
const SPACE_SHORT_NUMBER: &str = r#" [0-9]{1,3}"#;
const SHORT_DECIMAL_NUMBER: &str = r#"[0-9]{1,3}\.[0-9]"#;
const SPACE_SHORT_DECIMAL_NUMBER: &str = r#" [0-9]{1,3}\.[0-9]"#;
// Wrapped.
const WORD_WRAPPED_IN_BRACKETS: &str = r#"\[[A-Za-z]+\]"#;
const SHORT_NUMBER_WRAPPED_IN_BRACKETS: &str = r#"\[[0-9]{1,3}\]"#;
const WORD_WRAPPED_IN_QUOTES: &str = r#"['"][A-Za-z]+['"]"#;
const WORD_WRAPPED_IN_ANGLE_BRACKETS: &str = r#"<[A-Za-z]+>"#;
// Word punctuation.
const PUNCT_WORD: &str = r#"[[:punct:]][A-Za-z]+"#;
const SPACE_PUNCT_WORD: &str = r#" [[:punct:]][A-Za-z]+"#;
const WORD_PUNCT: &str = r#"[A-Za-z][[:punct:]]"#;
// Number punctuation.
const DOT_SHORT_NUMBER: &str = r#"\.[0-9]{1,3}"#;
// Whitespace.
const INDENT: &str = r#"(?:[ ]+)|[\t]+"#;
const NEWLINE_INDENT: &str = r#"(?:\n[ ]+)|(?:\n[\t]+)"#;
const WHITESPACE: &str = r#"\s+"#;
// Punctuation.
const SPACE_PUNCT_SPACE: &str = r#" ?[[:punct:]] ?"#;
const REPEATED_PUNCT: &str = r#"[[:punct:]]+"#;
const FEW_REPEATED_PUNCT: &str = r#"[[:punct:]]{1,4}"#;
const REPEATED_PUNCT_SPACE: &str = r#"(?: |[[:punct:]])+"#;
const FEW_REPEATED_PUNCT_SPACE: &str = r#"(?: |[[:punct:]]){1,4}"#;
const PUNCT_NEWLINE: &str = r#"[[:punct:]]+\n"#;
const REPEATED_PUNCT_NEWLINE_INDENT: &str = r#"[[:punct:]]+\n[ \t]+"#;

macro_rules! constexpr_regex {
    ($regex:expr) => {{
        fn generated_function() -> Regex {
            Regex::new($regex).unwrap()
        }
        generated_function as fn() -> Regex
    }};
}

macro_rules! space_anyof_space {
    ($arr:expr) => {{
        fn generated_function() -> Regex {
            let escaped = $arr
                .iter()
                .map(|&el| regex::escape(el))
                .collect::<Vec<String>>();

            Regex::new(&format!(
                r#" ?(?:{}) ?"#,
                escaped
                    .iter()
                    .map(|el| format!("(?:{})", el))
                    .collect::<Vec<String>>()
                    .join("|"),
            ))
            .unwrap()
        }

        generated_function as fn() -> Regex
    }};
}

const PACKAGE_KEYWORDS: &[&str] = &["package", "import", "export", "module", "use"];

const OPERATORS: &[&str] = &[
    "+", "-", "*", "/", "%", "&", "|", "^", "!", "~", "&&", "||", "==", "!=", "!==", "<", ">",
    "<=", ">=", "<<", ">>", ">>>", "++", "--", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
    "=>", "->", ".", "...", "?", "=", ":=", "[]", "()",
];

const CONTROL_FLOW_STATEMENTS: &[&str] = &[
    "if", "else", "for", "while", "do", "break", "continue", "return", "switch", "case", "default",
    "goto", "try", "catch", "finally", "throw", "assert", "yield", "defer", "await",
];

const LITERALS: &[&str] = &[
    "true",
    "false",
    "True",
    "False",
    "null",
    "nil",
    "None",
    "undefined",
];

const QUALIFIERS: &[&str] = &[
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

const PRIMITIVE_TYPES: &[&str] = &[
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
        &["hello"],
        &["Hello", "HELLO"],
    ),
    (
        "space-lowercase-word",
        constexpr_regex!(SPACE_LOWERCASE_WORD),
        &[" hello", " world"],
        &["Hello", " WORLD"],
    ),
    (
        "uppercase-word",
        constexpr_regex!(UPPERCASE_WORD),
        &["HELLO"],
        &["Hello", " WORLD"],
    ),
    (
        "space-uppercase-word",
        constexpr_regex!(SPACE_UPPERCASE_WORD),
        &[" HELLO", "WORLD"],
        &["Hello", " world"],
    ),
    (
        "capitalized-word",
        constexpr_regex!(CAPITALIZED_WORD),
        &["Hello"],
        &["HeLlO"],
    ),
    (
        "space-capitalized-word",
        constexpr_regex!(SPACE_CAPITALIZED_WORD),
        &[" Hello", "Hello"],
        &["HeLlO"],
    ),
    (
        "word",
        constexpr_regex!(WORD),
        &["hello", "Hello", "HELLO"],
        &["123"],
    ),
    (
        "space-word",
        constexpr_regex!(SPACE_WORD),
        &[" hello", " Hello", " HeLlO"],
        &["123"],
    ),
    (
        "space-english-word",
        constexpr_regex!(SPACE_ENGLISH_WORD),
        &["don't", " You'll", " He's"],
        &["ABC'DEF"],
    ),
    (
        "space-french-word",
        constexpr_regex!(SPACE_FRENCH_WORD),
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
    (
        "space-punct-word",
        constexpr_regex!(SPACE_PUNCT_WORD),
        &[" &abc", " :Abc", " +ABC"],
        &[],
    ),
    (
        "word-punct",
        constexpr_regex!(WORD_PUNCT),
        &["a&", "B:", "C+"],
        &[],
    ),
    // Number Punctuation
    (
        "dot-short-number",
        constexpr_regex!(DOT_SHORT_NUMBER),
        &[".1", ".123", ".789"],
        &[".1000"],
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
        "space-punct-space",
        constexpr_regex!(SPACE_PUNCT_SPACE),
        &[" # ", " ( ", " ) ", " { ", " } ", " != ", ", "],
        &[],
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
    // Code
    (
        "space-operator-space",
        space_anyof_space!(OPERATORS),
        &[" + ", " !=="],
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
