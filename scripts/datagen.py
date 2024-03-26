"""
Utility script to construct the pre-training dataset form The Stack v1.2
deduplicated dataset based on per-language quotas.
"""

import argparse
import glob
import os
import random
import re
import threading

import datasets


def mb(size: float) -> int:
    return int(size * (1024**2))


def has_many_non_chinese_non_ascii(content, p):
    total_chars = 0
    non_chinese_non_ascii_chars = 0

    for char in content:
        total_chars += 1

        if "\u0000" <= char <= "\u007f":
            continue

        if (
            "\u4e00" <= char <= "\u9fff"
            or "\u3400" <= char <= "\u4dbf"
            or "\uf900" <= char <= "\ufaff"
        ):
            continue

        non_chinese_non_ascii_chars += 1

    if total_chars == 0:
        return False
    portion = non_chinese_non_ascii_chars / total_chars

    return portion > p


def generate_the_stack(args, lang, quota):
    (train, test) = quota
    print(f"Generating ({train / mb(1)} MB, {test / mb(1)} MB) for {lang}")

    the_stack = datasets.load_dataset(
        "bigcode/the-stack-dedup",
        data_dir=f"data/{lang}",
        split="train",
        streaming=True,
    )

    for split in ["train", "test"]:
        os.makedirs(f"{args.output}/{split}", exist_ok=True)

    written = 0
    files = {
        split: open(f"{args.output}/{split}/{lang}.bin", "wb")
        for split in ["train", "test"]
    }

    regexes = [
        "[a-zA-Z0-9+/\\n=]{64,}",
        "(?:\\b(?:0x|\\\\x)?[0-9a-fA-F]{2}(?:,|\\b\\s*)){8,}",
        "(?:\\\\u[0-9a-fA-F]{4}){8,}",
    ]
    regexes = [re.compile(regex) for regex in regexes]

    size_filter = 0
    num_lines_filter = 0
    alphanum_filter = 0
    number_filter = 0
    regex_filter = 0
    unicode_filter = 0
    samples_visited = 0

    for sample in the_stack:
        samples_visited += 1
        (
            content,
            size,
            avg_line_length,
            max_line_length,
            alphanum_fraction,
        ) = (
            sample["content"],  # type: ignore
            sample["size"],  # type: ignore
            sample["avg_line_length"],  # type: ignore
            sample["max_line_length"],  # type: ignore
            sample["alphanum_fraction"],  # type: ignore
        )

        num_lines = content.count("\n")

        # Too short or too long
        if size < 16 or size > (mb(1) / 4):
            size_filter += 1
            continue

        # For JSON, YAML, TOML, we remove files with more than 512 lines
        # to minimize the impact of repeated tokens in data files.
        if lang in ["json", "yaml", "toml", "sql", "r", "hcl"]:
            if num_lines > 256:
                num_lines_filter += 1
                continue
        elif num_lines > 4096:
            num_lines_filter += 1

        # Suspicious line lengths
        if avg_line_length > 100 or avg_line_length < 10 or max_line_length > 1000:
            num_lines_filter += 1
            continue

        # Suspicious alphanum fraction
        if alphanum_fraction < 0.25:
            alphanum_filter += 1
            continue

        # Too many numbers (0123456789)
        if sum(char.isdigit() for char in content) > 0.3 * len(content):
            number_filter += 1
            continue

        # We remove the file if any of the substrings matching these expressions
        # is longer than 256 characters or if the fraction of matched characters
        # is more than 50% of the file.
        skip = False
        for regex in regexes:
            matches = regex.findall(content)
            if (
                any(len(match) > 256 for match in matches)
                or sum(len(match) for match in matches) / len(content) > 0.5
            ):
                skip = True
                break
        if skip:
            regex_filter += 1
            continue

        # Remove anything that's not Chinese or ASCII.
        if has_many_non_chinese_non_ascii(content, 0.2):
            unicode_filter += 1
            continue

        if written < test:
            f = files["test"]
        elif written < test + train:
            f = files["train"]
        else:
            break

        encoded = content.encode("utf-8")
        f.write(encoded)
        f.write(b"\0")
        written += len(encoded) + 1

    for f in files.values():
        f.close()

    print(
        f"Wrote {written}/{train + test} for {lang} to {args.output}. {samples_visited} samples visited. "
        f"Filters stats: size={size_filter} num_lines={num_lines_filter} alphanum={alphanum_filter} "
        f"number={number_filter} regex={regex_filter} unicode={unicode_filter}."
    )


def generate_chinese_markdown(args):
    train, test = map(
        lambda x: int(x * (1024**2)), map(float, args.chinese_markdown_quota.split(","))
    )

    print(f"Generating ({train / mb(1)} MB, {test / mb(1)} MB) for Chinese Markdown")

    chinese_markdown = datasets.load_dataset(
        "rojas-diego/chinese-markdown", split="train", streaming=True
    )

    for split in ["train" "test"]:
        os.makedirs(f"{args.output}/{split}", exist_ok=True)

    files = {
        split: open(f"{args.output}/{split}/chinese-markdown.bin", "wb")
        for split in ["train" "test"]
    }

    written = 0

    for sample in chinese_markdown:
        content = sample["code"]  # type: ignore

        if written < test:
            f = files["test"]
        elif written < test + train:
            f = files["train"]
        else:
            break

        encoded = content.encode("utf-8")
        f.write(encoded)
        f.write(b"\0")
        written += len(encoded) + 1

    for f in files.values():
        f.close()

    print(f"Wrote Chinese Markdown to {args.output}.")


def generate_infilling(args):
    train, test = map(
        lambda x: int(x * (1024**2)), map(float, args.infilling_quota.split(","))
    )

    print(f"Generating ({train / mb(1)} MB, {test / mb(1)} MB) for infilling")

    for split in ["train", "test"]:
        os.makedirs(f"{args.output}/{split}", exist_ok=True)

    files = {
        split: open(f"{args.output}/{split}/infilling.bin", "wb")
        for split in ["train", "test"]
    }

    written = 0

    inputs = glob.glob(f"{args.output}/train/*.bin")
    inputs = [open(input, "rb") for input in inputs]

    infilling = []
    for f in inputs:
        content = f.read()
        content = content.decode("utf-8")
        content = content.split("\0")
        infilling.extend(content[: len(content) // 5])
        f.close()

    while True:
        content = ""

        for _ in range(32):
            sample_idx = random.randrange(len(infilling))
            sample = infilling.pop(sample_idx)

            # Split the sample into chunks of characters
            chunk_size = max(32, min(len(sample) // 10, 128))
            chunks = [
                sample[i : i + chunk_size] for i in range(0, len(sample), chunk_size)
            ]

            if len(chunks) < 10:
                continue

            for _ in range(9):
                chunk_idx = random.randrange(len(chunks))
                chunk = chunks.pop(chunk_idx)
                content += chunk
                content += "\u007f"
            chunk_idx = random.randrange(len(chunks))
            content += chunks.pop(chunk_idx)

        if written < test:
            f = files["test"]
        elif written < test + train:
            f = files["train"]
        else:
            break

        encoded = content.encode("utf-8")
        f.write(encoded)
        f.write(b"\0")
        written += len(encoded) + 1

    for f in files.values():
        f.close()

    print(f"Wrote infilling to {args.output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="The directory in which to write the temporary files",
    )
    parser.add_argument(
        "--the-stack-quotas",
        type=str,
        nargs="+",
        help="The quotas for each language in The Stack in the form {lang}:{train_mb},{test_mb}",
    )
    parser.add_argument(
        "--chinese-markdown-quota",
        type=str,
        help="The quota for Chinese Markdown data in the form {train_mb},{test_mb}",
    )
    parser.add_argument(
        "--infilling-quota",
        type=str,
        help="The quota for infilling data in the form {train_mb},{test_mb}",
    )
    args = parser.parse_args()

    if args.the_stack_quotas:
        the_stack_quotas = [quota.split(":") for quota in args.the_stack_quotas]
        the_stack_quotas = [
            (lang, tuple(quota.split(","))) for [lang, quota] in the_stack_quotas
        ]
        the_stack_quotas = {
            lang: (mb(float(train_mb)), mb(float(test_mb)))
            for lang, (train_mb, test_mb) in the_stack_quotas
        }

        threads = []

        for lang, quota in the_stack_quotas.items():
            thread = threading.Thread(
                target=generate_the_stack, args=(args, lang, quota)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    if args.chinese_markdown_quota:
        generate_chinese_markdown(args)

    if args.infilling_quota:
        generate_infilling(args)
