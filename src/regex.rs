pub fn build_allow_regex<I>(regexes: I) -> String
where
    I: IntoIterator,
    I::Item: AsRef<str>,
{
    regexes
        .into_iter()
        .map(|r| format!("(?:^{}$)", r.as_ref()))
        .collect::<Vec<String>>()
        .join("|")
}

pub const SPLIT_REGEXES: &[(&str, &str)] = &[(
    "gpt4",
    r#"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"#,
)];

#[cfg(test)]
mod tests {
    // use std::collections::HashSet;
    // use regex::Regex;
    use fancy_regex::Regex as FancyRegex;

    use super::*;

    // #[test]
    // fn test_regexes() {
    //     for (name, regex, examples, counter_examples) in ALLOW_REGEXES {
    //         let re = Regex::new(format!("^{}$", regex).as_str()).unwrap();
    //         for &sample in examples.iter() {
    //             assert!(
    //                 re.is_match(sample),
    //                 "Rule {:?} expected to match {:?}",
    //                 name,
    //                 sample
    //             );
    //         }
    //         for &sample in counter_examples.iter() {
    //             assert!(
    //                 !re.is_match(sample),
    //                 "Rule {:?} expected not to match {:?}",
    //                 name,
    //                 sample
    //             );
    //         }
    //     }

    //     // Ensure there are no duplicate regex names
    //     let mut names = HashSet::new();
    //     let mut regexes = HashSet::new();
    //     for (name, regex, _, _) in ALLOW_REGEXES {
    //         assert!(names.insert(name), "Duplicate regex name found: {:?}", name);
    //         assert!(regexes.insert(regex), "Duplicate regex found: {:?}", regex);
    //     }
    // }

    #[test]
    fn test_gpt4_regexes() {
        for (_, regex) in SPLIT_REGEXES {
            let re = FancyRegex::new(regex).unwrap();
            let sample = "'./compone";

            for substring in re.find_iter(sample) {
                assert!(substring.is_ok());
                // Print the range of the substring
                let substring = substring.unwrap();
                println!("{:?}", substring.as_str());
                println!("{:?}", &sample[substring.range()]);
            }
        }
    }
}
