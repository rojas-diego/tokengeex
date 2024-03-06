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
    --output 'vocab.json' \
    --vocab-size 65536 \
    --train './hub/data/train/assembly.bin' \
    --train './hub/data/train/cuda.bin' \
    --train './hub/data/train/hcl.bin' \
    --train './hub/data/train/kotlin.bin' \
    --train './hub/data/train/php.bin' \
    --train './hub/data/train/shell.bin' \
    --train './hub/data/train/xml.bin' \
    --train './hub/data/train/c-sharp.bin' \
    --train './hub/data/train/dart.bin' \
    --train './hub/data/train/html.bin' \
    --train './hub/data/train/llvm.bin' \
    --train './hub/data/train/powershell.bin' \
    --train './hub/data/train/sql.bin' \
    --train './hub/data/train/yaml.bin' \
    --train './hub/data/train/c.bin' \
    --train './hub/data/train/diff.bin' \
    --train './hub/data/train/java.bin' \
    --train './hub/data/train/lua.bin' \
    --train './hub/data/train/python.bin' \
    --train './hub/data/train/swift.bin' \
    --train './hub/data/train/zig.bin' \
    --train './hub/data/train/chinese-markdown.bin' \
    --train './hub/data/train/dockerfile.bin' \
    --train './hub/data/train/javascript.bin' \
    --train './hub/data/train/makefile.bin' \
    --train './hub/data/train/r.bin' \
    --train './hub/data/train/tex.bin' \
    --train './hub/data/train/cmake.bin' \
    --train './hub/data/train/elixir.bin' \
    --train './hub/data/train/json.bin' \
    --train './hub/data/train/markdown.bin' \
    --train './hub/data/train/ruby.bin' \
    --train './hub/data/train/toml.bin' \
    --train './hub/data/train/cpp.bin' \
    --train './hub/data/train/go.bin' \
    --train './hub/data/train/jsx.bin' \
    --train './hub/data/train/pascal.bin' \
    --train './hub/data/train/rust.bin' \
    --train './hub/data/train/typescript.bin' \
    --train './hub/data/train/css.bin' \
    --train './hub/data/train/haskell.bin' \
    --train './hub/data/train/julia.bin' \
    --train './hub/data/train/perl.bin' \
    --train './hub/data/train/scala.bin' \
    --train './hub/data/train/vue.bin' \
    --valid './hub/data/valid/assembly.bin' \
    --valid './hub/data/valid/cuda.bin' \
    --valid './hub/data/valid/hcl.bin' \
    --valid './hub/data/valid/kotlin.bin' \
    --valid './hub/data/valid/php.bin' \
    --valid './hub/data/valid/shell.bin' \
    --valid './hub/data/valid/xml.bin' \
    --valid './hub/data/valid/c-sharp.bin' \
    --valid './hub/data/valid/dart.bin' \
    --valid './hub/data/valid/html.bin' \
    --valid './hub/data/valid/llvm.bin' \
    --valid './hub/data/valid/powershell.bin' \
    --valid './hub/data/valid/sql.bin' \
    --valid './hub/data/valid/yaml.bin' \
    --valid './hub/data/valid/c.bin' \
    --valid './hub/data/valid/diff.bin' \
    --valid './hub/data/valid/java.bin' \
    --valid './hub/data/valid/lua.bin' \
    --valid './hub/data/valid/python.bin' \
    --valid './hub/data/valid/swift.bin' \
    --valid './hub/data/valid/zig.bin' \
    --valid './hub/data/valid/chinese-markdown.bin' \
    --valid './hub/data/valid/dockerfile.bin' \
    --valid './hub/data/valid/javascript.bin' \
    --valid './hub/data/valid/makefile.bin' \
    --valid './hub/data/valid/r.bin' \
    --valid './hub/data/valid/tex.bin' \
    --valid './hub/data/valid/cmake.bin' \
    --valid './hub/data/valid/elixir.bin' \
    --valid './hub/data/valid/json.bin' \
    --valid './hub/data/valid/markdown.bin' \
    --valid './hub/data/valid/ruby.bin' \
    --valid './hub/data/valid/toml.bin' \
    --valid './hub/data/valid/cpp.bin' \
    --valid './hub/data/valid/go.bin' \
    --valid './hub/data/valid/jsx.bin' \
    --valid './hub/data/valid/pascal.bin' \
    --valid './hub/data/valid/rust.bin' \
    --valid './hub/data/valid/typescript.bin' \
    --valid './hub/data/valid/css.bin' \
    --valid './hub/data/valid/haskell.bin' \
    --valid './hub/data/valid/julia.bin' \
    --valid './hub/data/valid/perl.bin' \
    --valid './hub/data/valid/scala.bin' \
    --valid './hub/data/valid/vue.bin' \
    --test './hub/data/test/assembly.bin' \
    --test './hub/data/test/cuda.bin' \
    --test './hub/data/test/hcl.bin' \
    --test './hub/data/test/kotlin.bin' \
    --test './hub/data/test/php.bin' \
    --test './hub/data/test/shell.bin' \
    --test './hub/data/test/xml.bin' \
    --test './hub/data/test/c-sharp.bin' \
    --test './hub/data/test/dart.bin' \
    --test './hub/data/test/html.bin' \
    --test './hub/data/test/llvm.bin' \
    --test './hub/data/test/powershell.bin' \
    --test './hub/data/test/sql.bin' \
    --test './hub/data/test/yaml.bin' \
    --test './hub/data/test/c.bin' \
    --test './hub/data/test/diff.bin' \
    --test './hub/data/test/java.bin' \
    --test './hub/data/test/lua.bin' \
    --test './hub/data/test/python.bin' \
    --test './hub/data/test/swift.bin' \
    --test './hub/data/test/zig.bin' \
    --test './hub/data/test/chinese-markdown.bin' \
    --test './hub/data/test/dockerfile.bin' \
    --test './hub/data/test/javascript.bin' \
    --test './hub/data/test/makefile.bin' \
    --test './hub/data/test/r.bin' \
    --test './hub/data/test/tex.bin' \
    --test './hub/data/test/cmake.bin' \
    --test './hub/data/test/elixir.bin' \
    --test './hub/data/test/json.bin' \
    --test './hub/data/test/markdown.bin' \
    --test './hub/data/test/ruby.bin' \
    --test './hub/data/test/toml.bin' \
    --test './hub/data/test/cpp.bin' \
    --test './hub/data/test/go.bin' \
    --test './hub/data/test/jsx.bin' \
    --test './hub/data/test/pascal.bin' \
    --test './hub/data/test/rust.bin' \
    --test './hub/data/test/typescript.bin' \
    --test './hub/data/test/css.bin' \
    --test './hub/data/test/haskell.bin' \
    --test './hub/data/test/julia.bin' \
    --test './hub/data/test/perl.bin' \
    --test './hub/data/test/scala.bin' \
    --test './hub/data/test/vue.bin'
```
