# TokenGeeX - Efficient Tokenizer for CodeGeeX

This repository holds the code for the TokenGeeX Rust crate and Python package. TokenGeeX is an efficient tokenizer for code based on [UnigramLM (Taku Kudo 2018)](https://arxiv.org/abs/1804.10959) and [TokenMonster](https://github.com/alasdairforsythe/tokenmonster).

## Python

You can install the [PyPI TokenGeeX package](https://pypi.org/project/tokengeex/) through **pip**.

```bash
pip install tokengeex
```

Example usage:

```python
import tokengeex

tokenizer = tokengeex.load("unigram-32k.json")

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
    let tokenizer = tokengeex::load("unigram-32k.json").unwrap();

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

Example usage:

```bash
tokengeex train --model 'unigram' \
    --input 'data/train/en-cn-code-1GB.bin' \
    --output 'data/vocab/unigram-en-cn-code-32k.json' \
    --vocab-size '32768' \
    --max-token-length '24' \
    --max-words-per-token '3' \
    --max-sentence-size '64' \
    --shrinking-factor '0.8' \
    --initial-vocab-size '1000000' \
    --initial-vocab-cache 'data/cache/vocab-32k-en-cn-code-1GB' \
    --num-sub-iterations '2' \
    --suggested-tokens-file 'data/tokens/suggested.json' \
    --added-tokens-file 'data/tokens/added.json' \
    --special-token '<|CODE_PREFIX|>' \
    --special-token '<|CODE_SUFFIX|>' \
    --special-token '<|CODE_MIDDLE|>' \
    --special-token '<|EOS|>'
```
