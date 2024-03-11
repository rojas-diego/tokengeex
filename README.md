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

Here's the full command used to train vocabularies.

```shell
RUST_LOG=info TOKENGEEX_PARALLELISM=true RAYON_NUM_THREADS=8 tokengeex train --model 'unigram' \
    --processor nfc \
    --processor crlf \
    --processor capcode \
    --output 'tokenizer.json' \
    --vocab-size 65536 \
    --initial-vocab-size 1000000 \
    --initial-vocab-insert-probability 0.01 \
    --unigram-shrinking-factor 0.75 \
    --unigram-num-sub-iterations 2 \
    --max-token-length 24 \
    --allow '(?:^.$)|(?:^[[:punct:][:space:][DCU]]+$)|(?:^[\u3400-\u4DBF\u4E00-\u9FFF]+$)|(?:^(?:D?[UC]?)?(?: ?(?:(?:[a-z\._:/\-\*]+|[0-9]{1,4})(?:D?[UC]?))){0,4}$)|(?:^<D?[UC]? [a-z]+(?:>|/>| />)?$)' \
    --added-tokens-file ./hub/tokens/added.json \
    $(for lang in cpp; do echo "--train ./hub/data/train/${lang}.bin --valid ./hub/data/valid/${lang}.bin --test ./hub/data/test/${lang}.bin --suggested-tokens-file ./hub/tokens/suggested-${lang}.json"; done)
```

The full language list is:

```shell
$(for lang in assembly cuda hcl kotlin php shell xml c-sharp dart html llvm powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ./hub/data/train/${lang}.bin --valid ./hub/data/valid/${lang}.bin --test ./hub/data/test/${lang}.bin --suggested-tokens-file ./hub/tokens/suggested-${lang}.json"; done)
```

### Regexes

Here is an example set of Regexes used to influence the initial vocabulary.

| Regex                                                                     | Description                                                                                                                                                                                          | Example                               |
| ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| `^.$`                                                                     | Any lone Unicode character.                                                                                                                                                                          | `a`, `好`                             |
| `^D?[UC]? $`                                                              | Any capcode marker.                                                                                                                                                                                  | `DC `, `U `                           |
| `^ ?[0-9]{1,4}$`                                                          | Any number of 4 digits or less. May begin with a space.                                                                                                                                              | ` 123`, `9`                           |
| `^[\u3400-\u4DBF\u4E00-\u9FFF]+$`                                         | Any sequence of Chinese characters.                                                                                                                                                                  | `我叫罗杰斯`                          |
| `^ ?[a-z]+(?: [a-z]+){0,3}$`                                              | Any space separated sequence of up to 4 lowercase words. May begin with a space.                                                                                                                     | ` in order to`, `hello`               |
| `^(?:D?[UC]?)?(?: ?(?:(?:[a-z]+\|[0-9]{1,4})(?:D?[UC]?))){0,4}$`          | Any space separated sequence of up to 4 lowercase words. May begin with a space. Capcode and numbers allowed.                                                                                        | `DC complexDU casingD 123`            |
| `^(?:D?[UC]?)?(?: ?(?:(?:[a-z\._]+\|[0-9]{1,4})(?:D?[UC]?))){0,4}$`       | Any space separated sequence of up to 4 lowercase words. May begin with a space. Capcode and numbers allowed. Dots and underscores in-between words are allowed.                                     | ` users_D table`, `1.D 0`             |
| `^(?:D?[UC]?)?(?: ?(?:(?:[a-z\._:/\-\*]+\|[0-9]{1,4})(?:D?[UC]?))){0,4}$` | Any space separated sequence of up to 4 lowercase words. May begin with a space. Capcode and numbers allowed. Dots, underscores, colons, dashes, slashes, and aterisks in-between words are allowed. | `D https://D github.D com`            |
| `^<D?[UC]? [a-z]+(?:>\|/>\| />)?$`                                        | Any capcode-encoded XML/HTML tag, opened or closed.                                                                                                                                                  | `<D a>`, `<DU a`, `<D a/>`, `<D a />` |
| `^[[:punct:][:space:][DCU]]+$`                                            | Any sequence of punctuation, whitespace, and capcode.                                                                                                                                                | `\t`, `;\n\t\t`, `(D`                 |

### Configurations

#### Strict

Allows XML/HTML tags, sequences of up to three words (letters, capcode, numbers), Chinese words, Unicode characters.

```regexp
(?:^.$)|(?:^[[:punct:][:space:][DCU]]+$)|(?:^[\u3400-\u4DBF\u4E00-\u9FFF]+$)|(?:^(?:D?[UC]?)?(?: ?(?:(?:[a-z]+|[0-9]{1,4})(?:D?[UC]?))){0,4}$)|(?:^<D?[UC]? [a-z]+(?:>|/>| />)?$)
```

#### Base

Allows XML/HTML tags, complex sequences of up to four words (letters, capcode, numbers, underscores, dots), Chinese words, Unicode characters.

```regexp
(?:^.$)|(?:^[[:punct:][:space:][DCU]]+$)|(?:^[\u3400-\u4DBF\u4E00-\u9FFF]+$)|(?:^(?:D?[UC]?)?(?: ?(?:(?:[a-z\._]+|[0-9]{1,4})(?:D?[UC]?))){0,4}$)|(?:^<D?[UC]? [a-z]+(?:>|/>| />)?$)
```

#### Advanced

Allows XML/HTML tags, complex sequences of up to four words (letters, capcode, numbers, underscores, dots, slashes, colons, dashes, asteriks), Chinese words, Unicode characters.

```regexp
(?:^.$)|(?:^[[:punct:][:space:][DCU]]+$)|(?:^[\u3400-\u4DBF\u4E00-\u9FFF]+$)|(?:^(?:D?[UC]?)?(?: ?(?:(?:[a-z\._/:\-\*]+|[0-9]{1,4})(?:D?[UC]?))){0,4}$)|(?:^<D?[UC]? [a-z]+(?:>|/>| />)?$)
```
