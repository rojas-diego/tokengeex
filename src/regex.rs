use regex::Regex;

pub const IDIOMS: &[(&str, &str, &[&str], &[&str])] = &[
    // Char
    ("any-char", r#"."#, &["好", "A"], &[]),
    // Words
    (
        "lowercase-word",
        r#" ?[a-z]+"#,
        &["hello", " world"],
        &["Hello", "HELLO"],
    ),
    (
        "uppercase-word",
        r#" ?[A-Z]+"#,
        &["HELLO", " WORLD"],
        &["Hello", " WoRLD"],
    ),
    (
        "capitalized-word",
        r#" ?[A-Z][a-z]+"#,
        &[" Hello", "Hello"],
        &["HeLlO"],
    ),
    (
        "word",
        r#" ?[A-Za-z]+"#,
        &["hello", " Hello", " HeLlO"],
        &["123"],
    ),
    (
        "english-word",
        r#" ?[A-Za-z]+'[a-zA-Z]{1,2}"#,
        &["don't", " You'll", " He's"],
        &["ABC'DEF"],
    ),
    (
        "french-word",
        r#" ?[A-Za-zÀ-ÿ]+"#,
        &["Été", " compliqué"],
        &["مرحبا"],
    ),
    (
        "chinese-word",
        r#"[\u3400-\u4DBF\u4E00-\u9FFF]+"#,
        &["你好", "大家好"],
        &["مرحبا"],
    ),
    // Words Space
    (
        "lowercase-word-space",
        r#" ?[a-z]+ ?"#,
        &["hello ", " world "],
        &["Hello", "HELLO"],
    ),
    (
        "uppercase-word-space",
        r#" ?[A-Z]+ ?"#,
        &["HELLO ", " WORLD "],
        &["Hello", " WoRLD"],
    ),
    (
        "word-space",
        r#" ?[A-Za-z]+ ?"#,
        &["hello ", " Hello ", " HeLlO "],
        &["123"],
    ),
    // Multiple Words
    (
        "multi-lowercase-word",
        r#" ?[a-z]+(?: [a-z]+)+"#,
        &["hello world", " hello world"],
        &["Hello World", "HELLO WORLD"],
    ),
    (
        "multi-uppercase-word",
        r#" ?[A-Z]+(?: [A-Z]+)+"#,
        &["HELLO WORLD", " HELLO WORLD"],
        &["Hello World", "hello world"],
    ),
    (
        "multi-capitalized-word",
        r#" ?[A-Z][a-z]+(?: [A-Z][a-z]+)+"#,
        &[" Hello World", "Hello World"],
        &["HeLlO WoRLD"],
    ),
    (
        "multi-word",
        r#" ?[A-Za-z]+(?: [A-Za-z]+)+"#,
        &["hello world", " Hello World", " HeLlO WoRLD"],
        &["123"],
    ),
    // Grammar
    (
        "english-contraction",
        r#"'(?:re|ve|s|d|ll|t|m)"#,
        &["'re", "'ve", "'s", "'d", "'ll", "'t", "'m"],
        &[],
    ),
    // Numbers
    ("space-digit", r#" [0-9]"#, &[" 1", " 2", " 3"], &[" 10"]),
    (
        "short-number",
        r#"[0-9]{1,3}"#,
        &["1", "123", "789"],
        &["1000"],
    ),
    (
        "space-short-number",
        r#" [0-9]{1,3}"#,
        &[" 1", " 123", " 789"],
        &[],
    ),
    (
        "short-decimal-number",
        r#"[0-9]{1,3}\.[0-9]"#,
        &["1.1", "123.4", "789.9"],
        &["123.456", "1000.0"],
    ),
    (
        "space-short-decimal-number",
        r#" [0-9]{1,3}\.[0-9]"#,
        &[" 1.1", " 123.4", " 789.9"],
        &[" 123.456", " 1000.0"],
    ),
    // Wrapped
    (
        "word-wrapped-in-brackets",
        r#"\[[A-Za-z]+\]"#,
        &["[abc]", "[VALUE]"],
        &[],
    ),
    (
        "short-number-wrapped-in-brackets",
        r#"\[[0-9]{1,3}\]"#,
        &["[1]", "[123]", "[789]"],
        &[],
    ),
    (
        "word-wrapped-in-quotes",
        r#"['"][A-Za-z]+['"]"#,
        &["'abc'", "\"VALUE\""],
        &[],
    ),
    (
        "word-wrapped-in-angle-brackets",
        r#"<[A-Za-z]+>"#,
        &["<abc>", "<VALUE>"],
        &[],
    ),
    // Punctuation Word
    (
        "punct-word",
        r#"[[:punct:]][A-Za-z]+"#,
        &["&abc", ":Abc", "+ABC"],
        &[],
    ),
    // Space Punct Word
    (
        "space-punct-word",
        r#" [[:punct:]][A-Za-z]+"#,
        &[" &abc", " :Abc", " +ABC"],
        &[],
    ),
    // Word Punctuation
    (
        "word-punct",
        r#"[A-Za-z][[:punct:]]"#,
        &["a&", "B:", "C+"],
        &[],
    ),
    // Punctuation Number
    (
        "dot-short-number",
        r#"\.[0-9]{1,3}"#,
        &[".1", ".123", ".789"],
        &[".1000"],
    ),
    (
        "bracket-short-number",
        r#"\[[0-9]{1,3}"#,
        &["[1", "[123", "[789"],
        &["[1000"],
    ),
    // Snake Case
    (
        "snake-case",
        r#" ?[a-zA-Z]+(?:_[a-zA-Z]+)+"#,
        &["hello_world", "many_SUCH_words"],
        &["HelloWorld", "manySuchWords"],
    ),
    // URL
    (
        "scheme",
        r#"[a-z]+://"#,
        &["http://", "https://", "ftp://"],
        &[],
    ),
    (
        "port",
        r#":[0-9]{1,5}"#,
        &[":80", ":8080", ":443", ":65536"],
        &[],
    ),
    (
        "subdomains",
        r#"[a-z]+\.[a-z]+\.[a-z]+"#,
        &["www.google.com", "mail.google.com"],
        &[],
    ),
    // Path
    (
        "filename",
        r#"[A-Za-z]+\.[A-Za-z]+"#,
        &["file.txt", "file.html"],
        &[],
    ),
    (
        "path",
        r#"/[A-Za-z]+/[A-Za-z]+"#,
        &["/home/user", "/var/log"],
        &[],
    ),
    // Whitespace
    (
        "indent",
        r#"(?:[ ]+)|[\t]+"#,
        &[" ", "  ", "    ", "\t", "\t\t", "\t\t\t"],
        &["\t "],
    ),
    (
        "newline-indent",
        r#"\n[ ]+|[\t]+"#,
        &["\n ", "\n  ", "\n    ", "\n\t", "\n\t\t", "\n\t\t\t"],
        &["\n\t "],
    ),
    (
        "whitespace",
        r#"\s+"#,
        &[" ", "  ", "    ", "\n", "\n\n", "\t\t", " \n\t"],
        &[],
    ),
    // Punctuation
    (
        "repeated-punct",
        r#"[[:punct:]]+"#,
        &["####", "()[]{}"],
        &["\n#\n#\n#"],
    ),
    (
        "few-repeated-punct",
        r#"[[:punct:]]{1,4}"#,
        &["#", "##", "###", "()", "[]", "{}"],
        &["#####", "()[]{}"],
    ),
    (
        "repeated-punct-space",
        r#"(?: |[[:punct:]])+"#,
        &[" # ", " ( ", " ) ", " { ", " } ", " != ", ", "],
        &[],
    ),
    (
        "few-repeated-punct-space",
        r#"(?: |[[:punct:]]){1,4}"#,
        &[" # ", " ( ", " ) ", " { ", " } ", " != ", ", "],
        &[],
    ),
    (
        "punct-newline",
        r#"[[:punct:]]+\n"#,
        &[";\n", "]\n", "}\n"],
        &[";\n\n", "]\n\n", "}\n\n"],
    ),
    (
        "repeated-punct-newline-indent",
        r#"[[:punct:]]+\n[ \t]+"#,
        &[");\n\t\t", "]\n    "],
        &[],
    ),
    // C/C++
    (
        "cpp-pointer",
        r#" ?[A-Za-z]+(?:->)"#,
        &["value->", " ptr->"],
        &[],
    ),
    (
        "cpp-namespace-prefix",
        r#" ?[A-Za-z]+::"#,
        &["std::", " ns::"],
        &[],
    ),
    (
        "cpp-namespace-suffix",
        r#"::[A-Za-z]+"#,
        &["::std", "::ns"],
        &[],
    ),
    (
        "cpp-keywords",
        r#" ?(?:auto|char|const|double|float|int|long|short|signed|unsigned|void|volatile) ?"#,
        &["auto ", "char "],
        &[],
    ),
    (
        "cpp-preprocessor",
        r#"#(?:define|include|ifdef|ifndef|endif|if|else|elif|pragma) "#,
        &["#define ", "#include "],
        &[],
    ),
    ("cpp-include", r#"#include <"#, &["#include <"], &[]),
    // Go
    (
        "go-slice-primitive",
        r#" ?\[\]?(?:bool|int|int8|int16|int32|int64|uint|uint8|uint16|uint32|uint64|uintptr|float32|float64|string)"#,
        &["[]string", " []int"],
        &[],
    ),
    (
        "go-map-prefix-primitive",
        r#" ?map\[(?:bool|int|int8|int16|int32|int64|uint|uint8|uint16|uint32|uint64|uintptr|float32|float64|string)\]"#,
        &["map[string]", " map[int]"],
        &[],
    ),
    (
        "go-func",
        r#" ?func ?\(\)?"#,
        &["func (", " func ()", "func()"],
        &[],
    ),
    (
        "go-keywords",
        r#" ?(?:var|go|if|for|package|range|return|struct|type) ?"#,
        &["var ", "go "],
        &[],
    ),
    // Python
    (
        "dunder",
        r#" ?__[A-Za-z_0-9]+__"#,
        &["__init__", " __COMPLEX_123__"],
        &[],
    ),
    (
        "python-keywords",
        r#" ?(?:and|as|assert|break|class|continue|def|del|elif|else|except|finally|for|from|global|if|import|in|is|lambda|nonlocal|not|or|pass|raise|return|try|while|with|yield) ?"#,
        &["def ", "yield "],
        &[],
    ),
    // Rust
    (
        "rust-keywords",
        r#" ?(?:as|async|await|break|const|continue|crate|dyn|else|enum|extern|fn|for|if|impl|in|let|loop|match|mod|move|mut|pub|ref|return|self|Self|static|struct|super|trait|type|unsafe|use|where|while) ?"#,
        &["fn ", "use "],
        &[],
    ),
    // Java
    (
        "java-keywords",
        r#" ?(?:abstract|assert|boolean|break|byte|case|catch|char|class|const|continue|default|do|double|else|enum|extends|final|finally|float|for|goto|if|implements|import|instanceof|int|interface|long|native|new|null|package|private|protected|public|return|short|static|strictfp|super|switch|synchronized|this|throw|throws|transient|try|void|volatile|while) ?"#,
        &["class ", "void "],
        &[],
    ),
    // JavaScript
    (
        "js-keywords",
        r#" ?(?:break|case|catch|class|const|continue|debugger|default|delete|do|else|export|extends|finally|for|function|if|import|in|instanceof|new|return|super|switch|this|throw|try|typeof|var|void|while|with|yield) ?"#,
        &["class ", "yield "],
        &[],
    ),
    // TypeScript
    (
        "ts-keywords",
        r#" ?(?:abstract|as|break|case|catch|class|continue|const|constructor|debugger|declare|default|delete|do|else|enum|export|extends|false|finally|for|from|function|get|if|implements|import|in|infer|instanceof|interface|is|keyof|let|module|namespace|never|null|package|private|protected|public|readonly|require|global|return|set|static|string|super|switch|symbol|this|throw|true|try|type|typeof|undefined|var|void|while|with|yield) ?"#,
        &["class ", "yield "],
        &[],
    ),
    // HTML
    (
        "html-tag",
        r#"<[a-z]+ ?>?"#,
        &["<div>", "<span", "<div "],
        &["<DIV>", "<SPAN>"],
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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use regex::Regex;

    use super::*;

    #[test]
    fn test_regexes() {
        for (name, regex, examples, counter_examples) in IDIOMS {
            let re = Regex::new(format!("^{}$", regex).as_str()).unwrap();
            for &sample in examples.iter() {
                assert!(
                    re.is_match(sample),
                    "Rule {:?} expected to match {:?}",
                    name,
                    sample
                );
            }
            for &sample in counter_examples.iter() {
                assert!(
                    !re.is_match(sample),
                    "Rule {:?} expected not to match {:?}",
                    name,
                    sample
                );
            }
        }

        // Ensure there are no duplicate regex names
        let mut names = HashSet::new();
        let mut regexes = HashSet::new();
        for (name, regex, _, _) in IDIOMS {
            assert!(names.insert(name), "Duplicate regex name found: {:?}", name);
            assert!(regexes.insert(regex), "Duplicate regex found: {:?}", regex);
        }
    }
}
