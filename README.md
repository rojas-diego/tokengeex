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

Here's the full command used to train capcode vocabularies.

```shell
RUST_LOG=debug tokengeex train --model 'unigram' \
    --output 'capcode-65k.json' \
    --logfile 'capcode-65k.log' \
    --vocab-size 65536 \
    --processor 'nfc' \
    --processor 'crlf' \
    --processor 'capcode' \
    --initial-vocab-max-token-length 32 \
    --initial-vocab-size 5000000 \
    --initial-vocab-insert-probability 0.01 \
    --initial-vocab-allow '(?:^.$)|(?:^(?:(?:[[:punct:]]|(?:::))(?:(?:DU|DC|D) )(?:[a-z0-9]+))$)|(?:^(?:(?:[[:punct:] DCU]+)?(?:[[:space:]]*))$)|(?:^(?:[[:space:]]*(?:[[:punct:] DCU]+)?)$)|(?:^(?:^[\u3400-\u4DBF\u4E00-\u9FFF]+)$)|(?:^(?: (?:[a-z]+)://(?:(?:(?:(?:(?:(?:DU|DC|D) )(?:[a-z]+))(?:-(?:(?:(?:DU|DC|D) )(?:[a-z]+)))*)(?:\.(?:(?:(?:(?:DU|DC|D) )(?:[a-z]+))(?:-(?:(?:(?:DU|DC|D) )(?:[a-z]+)))*))*)|(?:(?:(?:DU|DC|D) )?(?:[0-9]+)(?:\.(?:(?:(?:DU|DC|D) )(?:[0-9]+))){3}))(?::(?:(?:DU|DC|D) )[0-9]{1,5})?)$)|(?:^(?:<D?[UC]? [a-z]+(?:>|/>| />)?)$)|(?:^(?:(?:(?:(?:(?:D|DU|DC|U|C) )| )?(?:[0-9]+))|(?:(?:(?:(?:D|DU|DC|U|C) )| )?(?:[a-z]+))){1,3}$)' \
    --unigram-shrinking-factor 0.8 \
    --unigram-num-sub-iterations 2 \
    --unigram-sample-regularization 'log' \
    --added-tokens-file './hub/tokens/capcode/added.json' \
    --suggested-tokens-file './hub/tokens/capcode/suggested.json' \
    $(for lang in assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin --test ${lang}:./hub/data/test/${lang}.bin --suggested-tokens-file ./hub/tokens/capcode/suggested-${lang}.json "; done)
```

Here's the full command used to train non-capcode vocabularies.

```shell
RUST_LOG=debug tokengeex train --model 'unigram' \
    --output '131k.json' \
    --logfile '131k.log' \
    --vocab-size 131072 \
    --processor 'nfc' \
    --processor 'crlf' \
    --processor 'capcode' \
    --initial-vocab-max-token-length 32 \
    --initial-vocab-size 10000000 \
    --initial-vocab-insert-probability 0.01 \
    --initial-vocab-allow '(?:^.$)|(?:^(?:[[:punct:]](?:(?:[A-Za-z]+)|(?:[0-9]{1,4})))$)|(?:^(?:(?:[ [:punct:]]+)?(?:[[:space:]]*))$)|(?:^(?:[[:space:]]*(?:[ [:punct:]]+)?)$)|(?:^(?:^[\u3400-\u4DBF\u4E00-\u9FFF]+)$)|(?:^(?:(?:[a-z]+)://(?:(?:(?:[A-Za-z]+)(?:-(?:[A-Za-z]+))*)(?:\.(?:(?:[A-Za-z]+)(?:-(?:[A-Za-z]+))*))*)(?:\.(?:[0-9]{1,3}(?:\.[0-9]{1,3}){3}))?(?::[0-9]{1,5})?)$)|(?:^(?:< ?[A-Za-z]+(?:>|/>| />)?)$)|(?:^(?:(?: ?(?:(?:(?:[0-9]{1,4})|(?:(?:[A-Za-z]+)(?:(?:[0-9]{1,4})(?:[A-Za-z]+))*(?:[0-9]{1,4})?)))){1,3})$)' \
    --unigram-shrinking-factor 0.8 \
    --unigram-num-sub-iterations 2 \
    --unigram-sample-regularization 'log' \
    --added-tokens-file './hub/tokens/base/added.json' \
    --suggested-tokens-file './hub/tokens/base/suggested.json' \
    $(for lang in assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin --test ${lang}:./hub/data/test/${lang}.bin --suggested-tokens-file ./hub/tokens/base/suggested-${lang}.json "; done)
```

### Configurations

#### Capcode

```regexp
(?:^.$)|(?:^(?:(?:[[:punct:]]|(?:::))(?:(?:DU|DC|D) )(?:[a-z0-9]+))$)|(?:^(?:(?:[[:punct:] DCU]+)?(?:[[:space:]]*))$)|(?:^(?:[[:space:]]*(?:[[:punct:] DCU]+)?)$)|(?:^(?:^[\u3400-\u4DBF\u4E00-\u9FFF]+)$)|(?:^(?: (?:[a-z]+)://(?:(?:(?:(?:(?:(?:DU|DC|D) )(?:[a-z]+))(?:-(?:(?:(?:DU|DC|D) )(?:[a-z]+)))*)(?:\.(?:(?:(?:(?:DU|DC|D) )(?:[a-z]+))(?:-(?:(?:(?:DU|DC|D) )(?:[a-z]+)))*))*)|(?:(?:(?:DU|DC|D) )?(?:[0-9]+)(?:\.(?:(?:(?:DU|DC|D) )(?:[0-9]+))){3}))(?::(?:(?:DU|DC|D) )[0-9]{1,5})?)$)|(?:^(?:(?:(?:D|DU|DC|U|C) )|(?: ?(?:[0-9]+))|(?: ?(?:[a-z]+)))+$)
```

#### No Capcode

```regexp
(?:^.$)|(?:^(?:[[:punct:]](?:(?:[A-Za-z]+)|(?:[0-9]{1,4})))$)|(?:^(?:(?:[ [:punct:]]+)?(?:[[:space:]]*))$)|(?:^(?:[[:space:]]*(?:[ [:punct:]]+)?)$)|(?:^(?:^[\u3400-\u4DBF\u4E00-\u9FFF]+)$)|(?:^(?:(?:[a-z]+)://(?:(?:(?:[A-Za-z]+)(?:-(?:[A-Za-z]+))*)(?:\.(?:(?:[A-Za-z]+)(?:-(?:[A-Za-z]+))*))*)(?:\.(?:[0-9]{1,3}(?:\.[0-9]{1,3}){3}))?(?::[0-9]{1,5})?)$)|(?:^(?:< ?[A-Za-z]+(?:>|/>| />)?)$)|(?:^(?:(?: ?(?:(?:(?:[0-9]{1,4})|(?:(?:[A-Za-z]+)(?:(?:[0-9]{1,4})(?:[A-Za-z]+))*(?:[0-9]{1,4})?)))){1,3})$)
```

### Evaluation

```shell
tokengeex evaluate -v ./hub/vocab/65k.json -l ./hub/log/65k.eval.log $(for lang in assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--test ${lang}:./hub/data/test/${lang}.bin "; done)
```
