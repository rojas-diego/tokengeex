# TokenGeeX Dataset

This document describes the composition of the training, validation and testing dataset of TokenGeeX.

## Composition

| Lang           | Train (GB) | Valid (MB) | Test (MB) |
| -------------- | ---------- | ---------- | --------- |
| Python         | 1.5        | 75         | 75        |
| C++            | 1          | 50         | 50        |
| C              | 1          | 50         | 50        |
| CSS            | 1          | 50         | 50        |
| Go             | 1          | 50         | 50        |
| HTML           | 1          | 50         | 50        |
| Java           | 1          | 50         | 50        |
| JavaScript     | 1          | 50         | 50        |
| Rust           | 1          | 50         | 50        |
| TypeScript     | 1          | 50         | 50        |
| C#             | 0.5        | 25         | 25        |
| PHP            | 0.5        | 25         | 25        |
| Ruby           | 0.5        | 25         | 25        |
| English Issues | 0.5        | 25         | 25        |
| Chinese Issues | 0.5        | 25         | 25        |
| Shell          | 0.2        | 10         | 10        |
| Vue            | 0.2        | 10         | 10        |
| Assembly       | 0.1        | 5          | 5         |
| Dockerfile     | 0.1        | 5          | 5         |
| Haskell        | 0.1        | 5          | 5         |
| Julia          | 0.1        | 5          | 5         |
| Lua            | 0.1        | 5          | 5         |
| PowerShell     | 0.1        | 5          | 5         |
| Scala          | 0.1        | 5          | 5         |
| SQL            | 0.1        | 5          | 5         |
| Tex            | 0.1        | 5          | 5         |
| JSON           | 0.1        | 5          | 5         |
| YAML           | 0.1        | 5          | 5         |
| Swift          | 0.1        | 5          | 5         |
| Scala          | 0.1        | 5          | 5         |
| Zig            | 0.1        | 5          | 5         |
| CUDA           | 0.05       | 1          | 1         |
| CMake          | 0.05       | 1          | 1         |
| Makefile       | 0.05       | 1          | 1         |

## Download

The dataset can be accessed through `storage.rojasdiego.com/tokengeex/data/{split}/{lang}.bin`. Each file is a NULL byte separated array of UTF-8 encoded files.
