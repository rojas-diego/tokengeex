# TokenGeeX Dataset

This document describes the composition of the training, validation and testing dataset of TokenGeeX.

## Composition

From [Chinese Markdown](https://huggingface.co/datasets/rojas-diego/chinese-markdown):

| Lang     | Train (GB) | Valid (MB) | Test (MB) |
| -------- | ---------- | ---------- | --------- |
| Markdown | 1          | 50         | 50        |

From [The Stack v1.2 (Deduplicated)](https://huggingface.co/datasets/bigcode/the-stack-dedup):

| Lang             | Train (GB) | Valid (MB) | Test (MB) |
| ---------------- | ---------- | ---------- | --------- |
| Markdown         | 1          | 50         | 50        |
| Python           | 1          | 50         | 50        |
| C++              | 1          | 50         | 50        |
| C                | 1          | 50         | 50        |
| CSS              | 1          | 50         | 50        |
| Go               | 1          | 50         | 50        |
| HTML             | 1          | 50         | 50        |
| Java             | 1          | 50         | 50        |
| JavaScript       | 1          | 50         | 50        |
| C#               | 0.5        | 25         | 25        |
| TypeScript       | 0.5        | 25         | 25        |
| Rust             | 0.5        | 25         | 25        |
| PHP              | 0.5        | 25         | 25        |
| Ruby             | 0.5        | 25         | 25        |
| Shell            | 0.5        | 25         | 25        |
| Vue              | 0.25       | 12.5       | 12.5      |
| JSON             | 0.25       | 12.5       | 12.5      |
| YAML             | 0.25       | 12.5       | 12.5      |
| Assembly         | 0.1        | 5          | 5         |
| Dockerfile       | 0.1        | 5          | 5         |
| Haskell          | 0.1        | 5          | 5         |
| Julia            | 0.1        | 5          | 5         |
| Lua              | 0.1        | 5          | 5         |
| Dart             | 0.1        | 5          | 5         |
| Kotlin           | 0.1        | 5          | 5         |
| PowerShell       | 0.1        | 5          | 5         |
| Scala            | 0.1        | 5          | 5         |
| Diff             | 0.1        | 5          | 5         |
| SQL              | 0.1        | 5          | 5         |
| JSX              | 0.1        | 5          | 5         |
| Tex              | 0.1        | 5          | 5         |
| Swift            | 0.1        | 5          | 5         |
| Scala            | 0.1        | 5          | 5         |
| Zig              | 0.1        | 5          | 5         |
| R                | 0.1        | 5          | 5         |
| XML              | 0.05       | 2.5        | 2.5       |
| Elixir           | 0.05       | 2.5        | 2.5       |
| Matlab           | 0.05       | 2.5        | 2.5       |
| LLVM             | 0.05       | 2.5        | 2.5       |
| Pascal           | 0.05       | 2.5        | 2.5       |
| Jupyter Notebook | 0.05       | 2.5        | 2.5       |
| Perl             | 0.05       | 2.5        | 2.5       |
| TOML             | 0.05       | 2.5        | 2.5       |
| HCL              | 0.05       | 2.5        | 2.5       |
| CUDA             | 0.05       | 2.5        | 2.5       |
| CMake            | 0.05       | 2.5        | 2.5       |
| Makefile         | 0.05       | 2.5        | 2.5       |

## Download

The dataset can be accessed through `storage.rojasdiego.com/tokengeex/data/{split}/{lang}.bin`. Each file is a NULL byte separated array of UTF-8 encoded files.
