# TokenGeeX - Efficient Tokenizer for CodeGeeX

This repository holds the code for the TokenGeeX Rust crate and Python package. TokenGeeX is a tokenizer for [CodeGeeX](https://github.com/THUDM/Codegeex2) aimed at code and Chinese. It is based on [UnigramLM (Taku Kudo 2018)](https://arxiv.org/abs/1804.10959).

## CLI

### Generate

```bash
RUST_LOG=debug tokengeex generate --output 'hub/vocab/v2/2500k-init.json' \
    --vocab-size 2500000 \
    --insert-probability 0.01 \
    --max-token-length 24 \
    --processor crlf \
    --processor nfc \
    --special-token '<|eos|>' \
    --special-token '<|suffix|>' \
    --special-token '<|prefix|>' \
    --special-token '<|middle|>' \
    --allow data/tokengeex.regex \
    $(for lang in infilling assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin "; done)
```

### Prune

```bash
RUST_LOG=debug tokengeex prune --input 'hub/vocab/v2/2500k-init.json' \
    --output 'hub/vocab/v2/65k-pruned.json' \
    --vocab-size 65536 \
    --shrink-factor 0.8 \
    --em-subiters 2 \
    $(for lang in infilling assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin:0.1 "; done)
```

### Filter

```bash
RUST_LOG=debug tokengeex filter --input 'hub/vocab/v2/65k-pruned.json' \
    --output 'hub/vocab/v2/50k-filtered.json' \
    --vocab-size 50000 \
    --min-score 13.0
```

### Merge

```bash
RUST_LOG=debug tokengeex merge --input 'hub/vocab/v2/50k-filtered.json' \
    --output 'hub/vocab/v2/51k.json' \
    --allow 'data/init.regex' \
    --num-merges 1000 \
    --step 10 \
    --scale-factor 0.9 \
    --max-token-length 24 \
    $(for lang in infilling assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin:0.1 "; done)
```

### Regex

#### Exact

The most restrictive pattern. Does not allow punctuation to be mixed in with words and strictly adheres to code structure. Does not allow words that mix casing. Digits are encoded as a single token.

```bash
$(for idiom in any-char lowercase-word uppercase-word capitalized-word english-contraction indent few-repeated-punct-space; do echo "-i ${idiom} "; done)
```

#### General

General-purpose pattern which is loosely analogous to GPT-4's pattern.

```bash
TODO!
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
