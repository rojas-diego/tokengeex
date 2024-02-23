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
print(tokenizer.token_to_id("token")) # 13513
print(tokenizer.id_to_token(13513)) # "token"

# Encode
ids = tokenizer.encode("def main(): print(\"Hello world!\")")
print(ids) # [68, 437, 12747, 58, 14653, 2807, 1735, 10120]

# Decode
print(tokenizer.decode(ids)) # "def main(): print(\"Hello world!\")"

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
    println!("{}", tokenizer.vocab_size()); // 32768
    println!("{}", .token_to_id("token").unwrap()) // 13513
    println!("{:?}", .id_to_token(13513).unwrap()) // "token"

    // Encode
    let ids = tokenizer.encode("def main(): print(\"Hello world!\")");
    println!("{:?}", ids); // [68, 437, 12747, 58, 14653, 2807, 1735, 10120]

    // Decode
    println!("{:?}", tokenizer.decode(ids)); // "def main(): print(\"Hello world!\")"

    // Byte fallbacks
    println!("{:?}", tokenizer.encode("电脑").map(|id| tokenizer.id_to_token(id))); // ["电", "<0xe8>", "<0x84>", "<0x91>"]
}
```

## CLI

You can install the [Rust binary crate](https://crates.io/crates/tokengeex) through **cargo**.

```
cargo install tokengeex --features cli
```

Here's a sample command to train a strict 512 vocabulary on ten megabytes of data.

```bash
RUST_LOG=info TOKENGEEX_PARALLELISM=true tokengeex train --model 'unigram' \
    --input 'data/train/code-10MB.bin' \
    --output 'data/vocab/code-512-strict.json' \
    --special-token '<|CODE_PREFIX|>' \
    --special-token '<|CODE_SUFFIX|>' \
    --special-token '<|CODE_MIDDLE|>' \
    --special-token '<|EOS|>' \
    --vocab-size 512 \
    --shrinking-factor '0.7' \
    --num-sub-iterations '2' \
    --vg-max-token-length '8' \
    --vg-max-words-per-token '2' \
    --vg-initial-vocab-size '2048' \
    --vg-insert-probability '0.1' \
    --vg-strict true
```

Here's a sample command to train a strict 4k vocabulary on a hundred megabytes of data.

```bash
RUST_LOG=info TOKENGEEX_PARALLELISM=true tokengeex train --model 'unigram' \
    --input 'data/train/code-100MB.bin' \
    --output 'data/vocab/code-4k-strict.json' \
    --special-token '<|CODE_PREFIX|>' \
    --special-token '<|CODE_SUFFIX|>' \
    --special-token '<|CODE_MIDDLE|>' \
    --special-token '<|EOS|>' \
    --vocab-size 4096 \
    --shrinking-factor '0.7' \
    --num-sub-iterations '2' \
    --suggested-tokens-file 'data/tokens/suggested.json' \
    --added-tokens-file 'data/tokens/added.json' \
    --vg-max-token-length '16' \
    --vg-max-words-per-token '3' \
    --vg-initial-vocab-size '100000' \
    --vg-insert-probability '0.01' \
    --vg-cache 'data/cache/code-100k-100MB-strict.json' \
    --vg-strict true
```

Here's a sample command to train a strict 16k vocabulary on a hundred megabytes of data.

```bash
RUST_LOG=info TOKENGEEX_PARALLELISM=true tokengeex train --model 'unigram' \
    --input 'data/train/code-100MB.bin' \
    --output 'data/vocab/code-16k-strict.json' \
    --special-token '<|CODE_PREFIX|>' \
    --special-token '<|CODE_SUFFIX|>' \
    --special-token '<|CODE_MIDDLE|>' \
    --special-token '<|EOS|>' \
    --vocab-size 16384 \
    --shrinking-factor '0.7' \
    --num-sub-iterations '2' \
    --suggested-tokens-file 'data/tokens/suggested.json' \
    --added-tokens-file 'data/tokens/added.json' \
    --vg-max-token-length '24' \
    --vg-max-words-per-token '3' \
    --vg-initial-vocab-size '1000000' \
    --vg-insert-probability '0.01' \
    --vg-cache 'data/cache/code-1000k-100MB-strict.json' \
    --vg-strict true
```
