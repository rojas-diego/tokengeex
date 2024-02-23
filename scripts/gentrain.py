"""
This script generates a .bin file which contain \0 separated sentences to train
TokenGeeX tokenizers on.
"""

import json
import sys

bytes = (int(sys.argv[1]) * 1024 * 1024) if len(sys.argv) > 1 else 100 * 1024 * 1024

lang = [
    "c++",
    "javascript",
    "python",
    "java",
    "go",
    "markdown",
    "rust",
]

files = [f"data/raw/{lang}.jsonl" for lang in lang]
files = [open(file, "r") for file in files]


def format_bytes(v):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if v < 1024:
            return f"{int(v)}{unit}"
        v /= 1024
    return f"{int(v)}PB"


with open(f"data/train/code-{format_bytes(bytes)}.bin", "wb") as f:
    bytes_written = 0

    while bytes_written < bytes:
        # Read one line of the JSONL file.
        for file in files:
            line = file.readline()
            if not line:
                sys.exit(1)
            sample = json.loads(line)
            encoded = sample["code"].encode("utf-8")
            f.write(encoded)
            f.write(b"\0")
            bytes_written += len(encoded) + 1
