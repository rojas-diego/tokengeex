# TokenGeeX - Efficient Tokenizer for CodeGeeX

This repository holds the code for the TokenGeeX Rust crate and Python package. TokenGeeX is a tokenizer for [CodeGeeX](https://github.com/THUDM/Codegeex2) aimed at code and Chinese. It is based on [UnigramLM (Taku Kudo 2018)](https://arxiv.org/abs/1804.10959).

## CLI

### Regex

```bash
RUST_LOG=debug tokengeex regex --output data/init.regex \
    -i word \
    -i english-word \
    -i french-word \
    -i chinese-word \
    -i english-contraction \
    -i space-digit \
    -i short-number \
    -i word-wrapped-in-brackets \
    -i short-number-wrapped-in-brackets \
    -i word-wrapped-in-quotes \
    -i word-wrapped-in-angle-brackets \
    -i punct-word \
    -i space-punct-word \
    -i dot-short-number \
    -i scheme \
    -i port \
    -i subdomains \
    -i filename \
    -i path \
    -i whitespace \
    -i punct-space \
    -i punct-newline-indent \
    -i cpp-pointer \
    -i cpp-namespace-prefix \
    -i cpp-namespace-suffix \
    -i cpp-preprocessor \
    -i cpp-include \
    -i go-slice-primitive \
    -i go-map-prefix-primitive \
    -i go-func \
    -i go-keywords \
    -i dunder \
    -i python-keywords \
    -i rust-keywords \
    -i js-keywords \
    -i ts-keywords \
    -i html-tag
```

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
    $(for lang in infilling assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin:0.01 "; done)
```
