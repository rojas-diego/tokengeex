# TokenGeeX Recipes

This document details how every single artifact inside `hub` was generated.

## Vocabularies

### `exact-500k-init`

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

### `exact-32k-pruned-high-dropout`

```bash
RUST_LOG=debug tokengeex prune --input 'hub/vocab/v2/exact-500k-init.json' \
    --output 'hub/vocab/v2/exact-32k-pruned-high-dropout.json' \
    --vocab-size 32000 \
    --dropout 0.1 \
    --shrink-factor 0.8 \
    --em-subiters 2 \
    $(for lang in assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin:0.25 "; done)
```

### `exact-32k-pruned-no-dropout`

```bash
RUST_LOG=debug tokengeex prune --input 'hub/vocab/v2/exact-500k-init.json' \
    --output 'hub/vocab/v2/exact-32k-pruned-no-dropout.json' \
    --vocab-size 32000 \
    --dropout 0.0 \
    --shrink-factor 0.8 \
    --em-subiters 2 \
    $(for lang in assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin:0.25 "; done)
```

### `exact-32k-pruned`

```bash
RUST_LOG=debug tokengeex prune --input 'hub/vocab/v2/exact-500k-init.json' \
    --output 'hub/vocab/v2/exact-32k-pruned.json' \
    --vocab-size 32000 \
    --dropout 0.05 \
    --shrink-factor 0.8 \
    --em-subiters 2 \
    $(for lang in assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin:0.25 "; done)
```

## Tokens

### Lang

Handwritten.

### Punct

Handwritten.

### Whitespace

```bash
RUST_LOG=debug tokengeex mine --output hub/tokens/whitespace.json \
    --num-idioms 100 \
    --pattern '[ ]{1,32}' \
    $(for lang in infilling assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin:0.25 "; done)
```
