# TokenGeeX Dataset

This document describes the composition of the training, validation and testing dataset of TokenGeeX.

## Composition

From [Chinese Markdown](https://huggingface.co/datasets/rojas-diego/chinese-markdown):

| Lang     | Train (GB) | Test (MB) |
| -------- | ---------- | --------- |
| Markdown | 1          | 100       |

From [The Stack v1.2 (Deduplicated)](https://huggingface.co/datasets/bigcode/the-stack-dedup):

| Lang       | Train (GB) | Test (MB) |
| ---------- | ---------- | --------- |
| Markdown   | 1          | 100       |
| Python     | 1          | 100       |
| C++        | 1          | 100       |
| C          | 1          | 100       |
| Go         | 1          | 100       |
| HTML       | 1          | 100       |
| Java       | 1          | 100       |
| JavaScript | 1          | 100       |
| CSS        | 0.5        | 50        |
| C#         | 0.5        | 50        |
| TypeScript | 0.5        | 50        |
| Rust       | 0.5        | 50        |
| PHP        | 0.5        | 50        |
| Ruby       | 0.5        | 50        |
| Shell      | 0.5        | 50        |
| Vue        | 0.25       | 25        |
| JSON       | 0.25       | 25        |
| YAML       | 0.25       | 25        |
| Dockerfile | 0.1        | 10        |
| Haskell    | 0.1        | 10        |
| Julia      | 0.1        | 10        |
| Lua        | 0.1        | 10        |
| Dart       | 0.1        | 10        |
| Kotlin     | 0.1        | 10        |
| PowerShell | 0.1        | 10        |
| Scala      | 0.1        | 10        |
| Diff       | 0.1        | 10        |
| SQL        | 0.1        | 10        |
| JSX        | 0.1        | 10        |
| Tex        | 0.1        | 10        |
| Swift      | 0.1        | 10        |
| Scala      | 0.1        | 10        |
| Zig        | 0.1        | 10        |
| R          | 0.1        | 10        |
| Assembly   | 0.05       | 5         |
| XML        | 0.05       | 5         |
| Elixir     | 0.05       | 5         |
| Pascal     | 0.05       | 5         |
| Perl       | 0.05       | 5         |
| TOML       | 0.05       | 5         |
| HCL        | 0.05       | 5         |
| CUDA       | 0.05       | 5         |
| CMake      | 0.05       | 5         |
| Makefile   | 0.05       | 5         |

## Download

The dataset can be accessed through `storage.rojasdiego.com/tokengeex/data/{split}/{lang}.bin`. Each file is a NULL byte separated array of UTF-8 encoded files.
