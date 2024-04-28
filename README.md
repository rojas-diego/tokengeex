# TokenGeeX - Efficient Tokenizer for CodeGeeX

This repository holds the code for the TokenGeeX Rust crate and Python package. TokenGeeX is a tokenizer for [CodeGeeX](https://github.com/THUDM/Codegeex2) aimed at code and Chinese. It is based on [UnigramLM (Taku Kudo 2018)](https://arxiv.org/abs/1804.10959).

## CLI

#### Exact

The most restrictive pattern. Does not allow punctuation to be mixed in with words and strictly adheres to code structure. Does not allow words that mix casing. Digits are encoded as a single token.

```bash
RUST_LOG=debug tokengeex regex --output data/exact.regex \
    $(for idiom in any-char lowercase-word uppercase-word capitalized-word english-contraction chinese-word indent few-repeated-punct-space; do echo "-i ${idiom} "; done)
```

#### Exact+

The pattern used for the merge step of exact vocabularies.

```bash
RUST_LOG=debug tokengeex regex --output data/exact-plus.regex \
    $(for idiom in any-char word english-word french-word chinese-word english-contraction punct-word newline-indent repeated-punct-space; do echo "-i ${idiom} "; done)
```

#### General

General-purpose pattern which is loosely analogous to GPT-4's pattern. Numbers of up to three digits are allowed.

```bash
RUST_LOG=debug tokengeex regex --output data/general.regex \
    $(for idiom in any-char word english-word french-word chinese-word english-contraction short-number punct-word newline-indent repeated-punct-space; do echo "-i ${idiom} "; done)
```

#### General+

The pattern used for the merge step of general vocabularies.

```bash
TODO!
```

#### Idiomatic

Permissive pattern which allows some common idioms to form. Allows multi-word tokens to form.

```bash
TODO!
```

#### Idiomatic+

The pattern used for the merge step of idiomatic vocabularies.

```bash
TODO!
```

#### Loose

Permits a wide range of patterns and idioms. Highest compression.

```bash
TODO!
```
