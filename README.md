# TokenGeeX - Efficient Tokenizer for CodeGeeX

This repository holds the code for the TokenGeeX Rust crate and Python package. TokenGeeX is a tokenizer for [CodeGeeX](https://github.com/THUDM/Codegeex2) aimed at code and Chinese. It is based on [UnigramLM (Taku Kudo 2018)](https://arxiv.org/abs/1804.10959).

## CLI

### Generate

```bash
RUST_LOG=debug tokengeex generate --output 'gpt4-gen-500k.json' \
    --vocab-size 500000 \
    --insert-probability 0.01 \
    --max-token-length 24 \
    --processor crlf \
    --processor nfc \
    --special-token '<|eos|>' \
    --special-token '<|suffix|>' \
    --special-token '<|prefix|>' \
    --special-token '<|middle|>' \
    --split data/gpt4.regex \
    $(for lang in infilling assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin:0.1 "; done)
```

### Prune

```bash
RUST_LOG=debug tokengeex prune --input 'gpt4-gen-500k.json' \
    --output 'gpt4-prune-50k.json' \
    --vocab-size 50000 \
    --shrink-factor 0.7 \
    --em-subiters 1 \
    $(for lang in infilling assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin:0.01 "; done)
```

### Filter

```bash
RUST_LOG=debug tokengeex filter --input 'gpt4-prune-50k.json' \
    --output 'gpt4-filtered-32k.json' \
    --vocab-size 32768 \
    --min-score 13.0
```
