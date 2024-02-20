import sys

from datasets import load_dataset as hf_load_dataset

LANGUAGES = set(
    [
        "python",
        "javascript",
        "typescript",
        "java",
        "go",
        "c++",
        "markdown",
    ]
)


the_stack_smol = hf_load_dataset("bigcode/the-stack-smol", split="train")

the_stack_smol = the_stack_smol.shuffle(seed=42)


bytes = int(sys.argv[1] if len(sys.argv) > 1 else 1024 * 1024 * 100)
bytes_generated = 0


def format_bytes(v):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if v < 1024:
            return f"{int(v)}{unit}"
        v /= 1024
    return f"{int(v)}PB"


with open(f"data/train/code-{format_bytes(bytes)}.bin", "wb") as f:
    for i, sample in enumerate(the_stack_smol):
        lang = sample["lang"].lower()
        avg_line_length = sample["avg_line_length"]
        max_line_length = sample["max_line_length"]
        alphanum_fraction = sample["alphanum_fraction"]

        if (
            lang not in LANGUAGES
            or max_line_length > 1000
            or alphanum_fraction < 0.5
            or avg_line_length < 10
            or avg_line_length > 100
        ):
            continue

        bytes_generated += len(sample["content"].encode("utf-8")) + 1

        content = sample["content"].encode("utf-8")

        f.write(content)

        if bytes_generated >= bytes:
            break
        else:
            f.write(b"\0")
