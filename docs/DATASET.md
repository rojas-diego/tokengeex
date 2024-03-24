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
| TypeScript | 1          | 100       |
| CSS        | 0.5        | 50        |
| C#         | 0.5        | 50        |
| Rust       | 0.5        | 50        |
| PHP        | 0.5        | 50        |
| Ruby       | 0.5        | 50        |
| Shell      | 0.5        | 50        |
| Vue        | 0.5        | 50        |
| JSX        | 0.5        | 50        |
| JSON       | 0.1        | 10        |
| YAML       | 0.1        | 10        |
| Dockerfile | 0.1        | 10        |
| Haskell    | 0.1        | 10        |
| Julia      | 0.1        | 10        |
| Lua        | 0.1        | 10        |
| Dart       | 0.1        | 10        |
| Kotlin     | 0.1        | 10        |
| PowerShell | 0.1        | 10        |
| Diff       | 0.1        | 10        |
| Swift      | 0.1        | 10        |
| Scala      | 0.1        | 10        |
| Zig        | 0.1        | 10        |
| R          | 0.05       | 5         |
| Tex        | 0.05       | 5         |
| SQL        | 0.05       | 5         |
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

Obtained from [The Stack v1.2 (Deduplicated)](https://huggingface.co/datasets/bigcode/the-stack-dedup):

| Lang      | Train (GB) | Test (MB) |
| --------- | ---------- | --------- |
| Infilling | 0.5        | 50        |

## Download

The dataset can be accessed through `storage.rojasdiego.com/tokengeex/data/{split}/{lang}.bin`. Each file is a NULL byte separated array of UTF-8 encoded files.

## Generate

```shell
python3 scripts/datagen.py --output ./hub/data --the-stack-quotas \
    markdown:1000,100 \
    python:1000,100 \
    cpp:1000,100 \
    c:1000,100 \
    go:1000,100 \
    html:1000,100 \
    java:1000,100 \
    javascript:1000,100 \
    typescript:1000,100 \
    jsx:500,50 \
    vue:500,50 \
    css:500,50 \
    c-sharp:500,50 \
    rust:500,50 \
    php:500,50 \
    ruby:500,50 \
    shell:500,50 \
    diff:100,10 \
    json:100,10 \
    yaml:100,10 \
    zig:100,10 \
    scala:100,10 \
    swift:100,10 \
    julia:100,10 \
    lua:100,10 \
    powershell:100,10 \
    kotlin:100,10 \
    dart:100,10 \
    haskell:100,10 \
    dockerfile:100,10 \
    r:100,10 \
    hcl:50,5 \
    toml:50,5 \
    xml:50,5 \
    sql:50,5 \
    tex:50,5 \
    cmake:50,5 \
    cuda:50,5 \
    perl:50,5 \
    pascal:50,5 \
    elixir:50,5 \
    assembly:50,5 \
    r:50,5 \
    makefile:50,5
```
