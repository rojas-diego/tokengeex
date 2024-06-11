# TokenGeeX - Efficient Tokenizer for CodeGeeX

This repository holds the code for the TokenGeeX Rust crate and Python package. TokenGeeX is a tokenizer for [CodeGeeX](https://github.com/THUDM/Codegeex2) aimed at code and Chinese. It is based on [UnigramLM (Taku Kudo 2018)](https://arxiv.org/abs/1804.10959).

## Regexes

TokenGeeX comes pre-packaged with a set of regexes which dictate what can be considered a token in the model's vocabulary during training. These regexes are used at vocabulary creation (using the `generate` command) and when extending a vocabulary using TokenGeeX's BPE implementation (using the `merge` command). Simply use the `--allow <file>` flag to specify the pattern you want to use.

You are free to create your own regexes. For this purpose, TokenGeeX includes a `regex` command to compile multiple regexes into a single file.

```
tokengeex regex
  Generate a Regex for downstream use with TokenGeeX.

  OPTIONS:
    -o, --output <output>
      Output file to save the Regex.

    -p, --pattern <pattern>
      Pattern to include. Can be either a named regex or a
      custom Regex.
```

For example, one can create a regex which matches any Unicode character and Chinese word.

```bash
tokengeex regex -p '.*' -p '[\u3400-\u4DBF\u4E00-\u9FFF]+' -o my-regex.txt
```

Thankfully, TokenGeeX ships with many pre-configured regexes (called "named patterns") which means you can write the following instead.

```bash
tokengeex regex -p any-char -p chinese-word -o my-regex.txt
```

To view the list of available named patterns, simply run.

```bash
tokengeex regex
```

Under `data/*.regex`, you can find many pre-built regular expressions that were used to train vocabularies. This section provides an explanation for them as well as the command used to generate them.

### Exact

The most restrictive pattern. Does not allow punctuation to be mixed in with words and strictly adheres to code structure. Does not allow words that mix casing. Digits are encoded as a single token.

```bash
RUST_LOG=debug tokengeex regex --output data/exact.regex \
    $(for pattern in any-char lowercase-word uppercase-word capitalized-word english-contraction chinese-word indent space-operator-space space-punct-space; do echo "--pattern ${pattern} "; done)
```

### Fine

A more permissive pattern than "exact". Words are allowed to be prepended with a space.

```bash
RUST_LOG=debug tokengeex regex --output data/fine.regex \
    $(for pattern in any-char lowercase-word uppercase-word capitalized-word english-contraction chinese-word indent space-operator-space space-punct-space; do echo "--pattern ${pattern} "; done)
```

### General

A pattern analogous to GPT-4's regex. Words can be prepended with punctuation. Numbers of up to three digits are allowed.

### Coarse

```bash
```

### Idiomatic

## Mine

TokenGeeX lets you mine large-scale datasets for common idioms.

```
tokengeex mine
  Mine for common idioms from a large scale dataset.

  OPTIONS:
    -n, --num-idioms <num_idioms>
      Number of idioms to keep from the set of most occuring
      idioms.

    -o, --output <output>
      Output file to save the idioms.

    --train <input>
      List of source files to train the tokenizer on. Must be
      formatted according to {name}:{path}[:proportion].

    -p, --pattern <pattern>
      Pattern to look for. Can be either a named regex or a
      custom regex.
```

For example, you can search for the most frequent Chinese words.

```bash
RUST_LOG=debug tokengeex mine --output idioms.json \
    --num-idioms 100 \
    --pattern chinese-word \
    $(for lang in infilling assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin:0.25 "; done)
```

## Generate

To generate a large vocabulary to prune later, you can use the `generate` subcommand.

```
tokengeex generate
  Create a new tokenizer with a vocabulary generated from a large
  training dataset.

  OPTIONS:
    -v, --vocab-size <vocab_size>
      The size of the vocabulary to generate.

    -o, --output <output>
      The output file to save the tokenizer.

    --processor <processor>
      Apply a processor to the input data.

    --train <input>
      List of source files to train the tokenizer on. Must be
      formatted according to {name}:{path}[:proportion].

    --special <special>
      Special token.

    --suggested <suggested>
      Path to a file which contains an array of suggested tokens.

    --added <added>
      Path to a file which contains an array of added tokens.

    --allow <allow>
      Path to a file which contains a regular expression. Only
      merges that match this regex will be considered.

    --split <split>
      Path to a file which contains a regular expression. If
      specified, every sample will be split according to this
      regex before being processed. Supports fancy regex syntax.

    --insert-probability <insert_probability>
      Probability of inserting a new token to the vocabulary.

    --max-token-length <max_token_length>
      Maximum token length.
```

For example, you can generate a vocabulary using the exact regex.

```bash
RUST_LOG=debug tokengeex generate --output 'hub/vocab/v2/exact-500k-init.json' \
    --vocab-size 500000 \
    --insert-probability 0.01 \
    --max-token-length 16 \
    --processor crlf \
    --processor nfc \
    --special '<|eos|>' \
    --special '<|pad|>' \
    --special '<|lang|>' \
    --special '<|filename|>' \
    --special '<|suffix|>' \
    --special '<|prefix|>' \
    --special '<|middle|>' \
    --added hub/tokens/lang.json \
    --added hub/tokens/whitespace.json \
    --suggested hub/tokens/punct.json \
    --allow data/exact.regex \
    $(for lang in infilling assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin "; done)
```

## Prune

To optimise large vocabularies, you can prune the vocabulary using the `prune` command.

```
tokengeex prune
  Iteratively prune the vocabulary by removing the least frequent
  tokens.

  OPTIONS:
    -i, --input <input>
      The input tokenizer file.

    -o, --output <output>
      The output tokenizer file.

    -v, --vocab-size <vocab_size>
      The size of the vocabulary to prune to.

    --train <input>
      List of source files to train the tokenizer on. Must be
      formatted according to {name}:{path}[:proportion].

    --dropout <dropout>
      Dropout factor. The probability of omitting a token from
      the segmentation of a sample.

    --shrink-factor <shrink_factor>
      How much to shrink the vocabulary at each iteration.

    --em-subiters <em_subiters>
      Number of sub-iterations for the EM algorithm.
```

For example, we can reduce our 500,000 entries vocabularies to 32,000 entries.

```bash
RUST_LOG=debug tokengeex prune --input 'hub/vocab/v2/exact-500k-init.json' \
    --output 'hub/vocab/v2/exact-32k-pruned-high-dropout.json' \
    --vocab-size 32000 \
    --dropout 0.1 \
    --shrink-factor 0.8 \
    --em-subiters 2 \
    $(for lang in assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin:0.25 "; done)
```

## Filter

To remove low-frequency tokens from a vocabulary, you can use the `filter` subcommand.

The following command will remove all tokens with a log probability lower than `-13.0` until the vocabulary size is 30,000.

```bash
RUST_LOG=debug tokengeex filter --input 'hub/vocab/v2/exact-32k-pruned.json' \
    --output 'hub/vocab/v2/exact-30k-filtered.json' \
    --vocab-size 30000 \
    --min-score -13.0
```

## Merge

To merge commonly occuring tokens into a single token, you can use the `merge` subcommand.

```bash
RUST_LOG=debug tokengeex merge --input 'hub/vocab/v2/exact-30k-filtered.json' \
    --output 'hub/vocab/v2/exact-32k-merged.json' \
    --allow 'data/fine.regex' \
    --num-merges 2000 \
    --step 100 \
    --scale-factor 0.9 \
    --max-token-length 20 \
    $(for lang in assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin:0.25 "; done)
```

## Evaluate

You can evaluate the compression and token frequency distribution of TokenGeeX vocabularies but also of third-party libraries like SentencePiece, HuggingFace Transformers, HuggingFace Tokenizers, and TikToken.

```bash
python scripts/evaluate.py -f hub/vocab/v2/exact-32k-pruned.json -i './hub/data/test/*.bin' -o hub/eval/v2/exact-32k-pruned.json -l tokengeex
```

You can then plot the results of the evaluation with the following command.

```bash
python scripts/plot.py -i hub/eval/v2/exact-32k-pruned.json &
```
