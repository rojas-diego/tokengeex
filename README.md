# TokenGeeX - Efficient Tokenizer for CodeGeeX

This repository holds the code for the TokenGeeX Rust crate and Python package. TokenGeeX is a tokenizer for [CodeGeeX](https://github.com/THUDM/Codegeex2) aimed at code and Chinese. It is based on [UnigramLM (Taku Kudo 2018)](https://arxiv.org/abs/1804.10959).

## CLI

### Regex

```bash
RUST_LOG=debug tokengeex regex --output data/exact.regex \
    # Place the idioms and regexes here.
```

#### Exact

The most restrictive pattern. Does not allow punctuation to be mixed in with words and strictly adheres to code structure. Does not allow words that mix casing. Digits are encoded as a single token.

```bash
$(for idiom in any-char lowercase-word uppercase-word capitalized-word english-contraction chinese-word indent few-repeated-punct-space; do echo "-i ${idiom} "; done)
```

#### General

General-purpose pattern which is loosely analogous to GPT-4's pattern. Numbers of up to three digits are allowed.

```bash
$(for idiom in any-char ch; do echo "-i ${idiom} "; done)
```

#### Broad

Permissive pattern which allows some common idioms to form. Allows multi-word tokens to form.

```bash
TODO!
```

#### Loose

Permits a wide range of patterns and idioms. Highest compression.

```bash
TODO!
```
