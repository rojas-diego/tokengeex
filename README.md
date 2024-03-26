# TokenGeeX - Efficient Tokenizer for CodeGeeX

This repository holds the code for the TokenGeeX Rust crate and Python package. TokenGeeX is a tokenizer for [CodeGeeX](https://github.com/THUDM/Codegeex2) aimed at code and Chinese. It is based on [UnigramLM (Taku Kudo 2018)](https://arxiv.org/abs/1804.10959) and [TokenMonster](https://github.com/alasdairforsythe/tokenmonster).

## Python

You can install the [PyPI TokenGeeX package](https://pypi.org/project/tokengeex/) through **pip**.

```bash
pip install tokengeex
```

Example usage:

```python
import tokengeex

tokenizer = tokengeex.load("code-32k-strict.json")

# Vocab
print(tokenizer.vocab_size()) # 32768
print(tokenizer.token_to_id(b"token")) # 13513
print(tokenizer.id_to_token(13513)) # (b"token", -13.322)

# Encode
ids = tokenizer.encode("def main(): print(\"Hello world!\")")
print(ids) # [68, 437, 12747, 58, 14653, 2807, 1735, 10120]

# Decode
print(tokenizer.decode(ids, include_special_tokens=False)) # "def main(): print(\"Hello world!\")"

# Byte fallbacks
print([tokenizer.id_to_token(id) for id in tokenizer.encode("电脑")]) # ["电", "<0xe8>", "<0x84>", "<0x91>"]
```

## Rust

You can install the [Rust library crate](https://crates.io/crates/tokengeex) through **cargo**.

```bash
cargo add tokengeex
```

Example usage:

```rust
fn main() {
    let tokenizer = tokengeex::load("code-32k-strict.json").unwrap();

    // Vocab
    println!("{}", tokenizer.vocab_size());
    println!("{}", tokenizer.token_to_id("token").unwrap())
    println!("{:?}", tokenizer.id_to_token(13513).unwrap())

    // Encode
    let ids = tokenizer.encode("def main(): print(\"Hello world!\")");
    println!("{:?}", ids); // [68, 437, 12747, 58, 14653, 2807, 1735, 10120]

    // Decode
    println!("{:?}", tokenizer.decode(ids, false)); // "def main(): print(\"Hello world!\")"

    // Byte fallbacks
    println!("{:?}", tokenizer.encode("电脑").map(|id| tokenizer.id_to_token(id))); // ["电", "<0xe8>", "<0x84>", "<0x91>"]
}
```

## CLI

### Train

You can install the [Rust binary crate](https://crates.io/crates/tokengeex) through **cargo**.

```
cargo install tokengeex --features cli
```

Here's the full command used to train base vocabularies.

```shell
RUST_LOG=debug RAYON_NUM_THREADS=120 tokengeex train \
    --model 'unigram' \
    --output 'base-131k.json' \
    --logfile 'base-131k.log' \
    --vocab-size 131072 \
    --processor 'nfc' \
    --processor 'crlf' \
    --initial-vocab-max-token-length 32 \
    --initial-vocab-size 10000000 \
    --initial-vocab-insert-probability 0.01 \
    --initial-vocab-allow "$(cat data/base.regex)" \
    --unigram-shrinking-factor 0.8 \
    --unigram-num-sub-iterations 2 \
    --unigram-sample-regularization 'log' \
    --added-tokens-file './hub/tokens/base/added.json' \
    --suggested-tokens-file './hub/tokens/base/suggested.json' \
    $(for lang in infilling assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin --test ${lang}:./hub/data/test/${lang}.bin --suggested-tokens-file ./hub/tokens/base/suggested-${lang}.json "; done)
```

Here's the full command used to train capcode vocabularies.

```shell
RUST_LOG=debug RAYON_NUM_THREADS=120 tokengeex train \
    --model 'unigram' \
    --output 'capcode-131k.json' \
    --logfile 'capcode-131k.log' \
    --vocab-size 131072 \
    --processor 'nfc' \
    --processor 'crlf' \
    --processor 'capcode' \
    --initial-vocab-max-token-length 32 \
    --initial-vocab-size 10000000 \
    --initial-vocab-insert-probability 0.01 \
    --initial-vocab-allow "$(cat data/capcode.regex)" \
    --unigram-shrinking-factor 0.8 \
    --unigram-num-sub-iterations 2 \
    --unigram-sample-regularization 'log' \
    --added-tokens-file './hub/tokens/capcode/added.json' \
    --suggested-tokens-file './hub/tokens/capcode/suggested.json' \
    $(for lang in infilling assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin --test ${lang}:./hub/data/test/${lang}.bin --suggested-tokens-file ./hub/tokens/capcode/suggested-${lang}.json "; done)
```

### Extend with BPE

```shell
RUST_LOG=debug RAYON_NUM_THREADS=120 tokengeex bpe \
    --output ./capcode-131k-extended.json \
    --vocab ./capcode-131k.json \
    --num-merges 10000 \
    --step 100 \
    --score-scale-factor 0.75 \
    --max-merge-length 12 \
    --ignore '^$' \
    $(for lang in infilling assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin --test ${lang}:./hub/data/test/${lang}.bin "; done)
```
