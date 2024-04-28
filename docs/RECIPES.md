# TokenGeeX Recipes

## `exact-500k-init`

```bash
RUST_LOG=debug tokengeex generate --output 'hub/vocab/v2/exact-500k-init.json' \
    --vocab-size 500000 \
    --insert-probability 0.01 \
    --max-token-length 12 \
    --processor crlf \
    --processor nfc \
    --special-token '<|eos|>' \
    --special-token '<|suffix|>' \
    --special-token '<|prefix|>' \
    --special-token '<|middle|>' \
    --allow data/exact.regex \
    $(for lang in infilling assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin "; done)
```

## `exact-32k-pruned`

```bash
RUST_LOG=debug tokengeex prune --input 'hub/vocab/v2/exact-500k-init.json' \
    --output 'hub/vocab/v2/exact-32k-pruned.json' \
    --vocab-size 32000 \
    --shrink-factor 0.8 \
    --em-subiters 2 \
    $(for lang in infilling assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin:0.25 "; done)
```

## `exact-30k-filtered`

```bash
RUST_LOG=debug tokengeex filter --input 'hub/vocab/v2/exact-32k-pruned.json' \
    --output 'hub/vocab/v2/exact-30k-filtered.json' \
    --vocab-size 30000 \
    --min-score 13.0
```

## `exact-32k-merged`

```bash
RUST_LOG=info tokengeex merge --input 'hub/vocab/v2/exact-30k-filtered.json' \
    --output 'hub/vocab/v2/exact-32k-merged.json' \
    --allow 'data/exact-plus.regex' \
    --num-merges 2000 \
    --step 100 \
    --scale-factor 0.9 \
    --max-token-length 20 \
    $(for lang in infilling assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin:0.25 "; done)
```

## `general-1000k-init`

```bash
RUST_LOG=debug tokengeex generate --output 'hub/vocab/v2/general-1000k-init.json' \
    --vocab-size 1000000 \
    --insert-probability 0.01 \
    --max-token-length 16 \
    --processor crlf \
    --processor nfc \
    --special-token '<|eos|>' \
    --special-token '<|suffix|>' \
    --special-token '<|prefix|>' \
    --special-token '<|middle|>' \
    --allow data/general.regex \
    $(for lang in infilling assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin "; done)
```

## `general-32k-pruned`

```bash
RUST_LOG=debug tokengeex prune --input 'hub/vocab/v2/general-1000k-init.json' \
    --output 'hub/vocab/v2/general-32k-pruned.json' \
    --vocab-size 32000 \
    --shrink-factor 0.8 \
    --em-subiters 2 \
    $(for lang in infilling assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin:0.25 "; done)
```

## `general-30k-filtered`

```bash
RUST_LOG=debug tokengeex filter --input 'hub/vocab/v2/general-32k-pruned.json' \
    --output 'hub/vocab/v2/general-30k-filtered.json' \
    --vocab-size 30000 \
    --min-score 13.0
```

## `general-32k-merged`

```bash
RUST_LOG=debug tokengeex merge --input 'hub/vocab/v2/general-30k-filtered.json' \
    --output 'hub/vocab/v2/general-32k-merged.json' \
    --allow 'data/idiomatic.regex' \
    --num-merges 2000 \
    --step 100 \
    --scale-factor 0.9 \
    --max-token-length 20 \
    $(for lang in infilling assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin:0.25 "; done)
```
