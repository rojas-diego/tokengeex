# TokenGeeX Regexes

TokenGeeX comes pre-packaged with a set of regular expressions which dictate what can be considered a token in the model's vocabulary during training. These regexes are used at vocabulary creation (using the `generate` command) and when extending a vocabulary using TokenGeeX's BPE implementation (using the `merge` command). Simply use the `--allow <file>` flag to specify the Regex you want to use.

You are free to create your own Regexes. For this purpose, TokenGeeX includes a `regex` command to compile multiple Regexes into a file.

```
tokengeex regex
  Generate a Regex for downstream use with TokenGeeX.

  OPTIONS:
    -o, --output <output>
      Output file to save the Regex.

    -i, --idiom <idiom>
      Comma separated list of idioms to use.

    -r, --rule <rule>
      List of Regex rules to use in addition to the idioms.
```

For example, one can create a regex which matches any Unicode character and Chinese word.

```bash
tokengeex regex -r '.*' -r '[\u3400-\u4DBF\u4E00-\u9FFF]+' -o my-regex.txt
```

Thankfully, TokenGeeX ships with many common idioms which means you can write the following instead.

```bash
tokengeex regex -i any-char -i chinese-word -o my-regex.txt
```

To view the list of available idioms, simply run.

```bash
tokengeex regex
```

## Pre-Built Regexes

Under `data/*.regex`, you can find many pre-built regular expressions that were used to train vocabularies. This section provides an explanation for them as well as the command used to generate them.

### Exact

The most restrictive pattern. Does not allow punctuation to be mixed in with words and strictly adheres to code structure. Does not allow words that mix casing. Digits are encoded as a single token.

```bash
RUST_LOG=debug tokengeex regex --output data/exact.regex \
    $(for idiom in any-char lowercase-word uppercase-word capitalized-word english-contraction chinese-word indent few-repeated-punct-space; do echo "-i ${idiom} "; done)
```

### Exact+

The pattern used for the merge step of exact vocabularies.

```bash
RUST_LOG=debug tokengeex regex --output data/exact-plus.regex \
    $(for idiom in any-char word english-word french-word chinese-word english-contraction punct-word newline-indent repeated-punct-space; do echo "-i ${idiom} "; done)
```

### General

General-purpose pattern which is loosely analogous to GPT-4's pattern. Numbers of up to three digits are allowed.

```bash
RUST_LOG=debug tokengeex regex --output data/general.regex \
    $(for idiom in any-char word english-word french-word chinese-word english-contraction short-number punct-word newline-indent repeated-punct-space; do echo "-i ${idiom} "; done)
```

### General+

The pattern used for the merge step of general vocabularies.

```bash
TODO!
```

### Idiomatic

Permissive pattern which allows some common idioms to form. Allows multi-word tokens to form.

```bash
TODO!
```

### Idiomatic+

The pattern used for the merge step of idiomatic vocabularies.

```bash
TODO!
```

### Loose

Permits a wide range of patterns and idioms. Highest compression.

```bash
TODO!
```
