"""
Utility script to construct the pre-training dataset form The Stack v1.2
deduplicated dataset based on per-language quotas.
"""

import argparse
import os
import threading

import datasets


def mb(size: float) -> int:
    return int(size * (1024**2))


def generate_the_stack(args, lang, quota):
    (train, valid, test) = quota
    print(
        f"Generating ({train / mb(1)} MB, {valid / mb(1)} MB, {test / mb(1)} MB) for {lang}"
    )

    the_stack = datasets.load_dataset(
        "bigcode/the-stack-dedup",
        data_dir=f"data/{lang}",
        split="train",
        streaming=True,
    )

    for split in ["train", "valid", "test"]:
        os.makedirs(f"{args.output}/{split}", exist_ok=True)

    written = 0
    files = [
        open(f"{args.output}/{split}/{lang}.bin", "wb")
        for split in ["train", "valid", "test"]
    ]

    for sample in the_stack:
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

        # Languages for which we may not have enough data.
        if lang not in [
            "cuda",
            "cmake",
            "llvm",
            "matlab",
            "nginx",
            "elixir",
            "jupyter-notebook",
            "perl",
            "toml",
            "hcl",
            "makefile",
        ]:
            if (
                size > mb(1)
                or avg_line_length > 100
                or alphanum_fraction < 0.25
                or (max_line_length > 1000 and lang != "json" and lang != "html")
            ):
                continue

        if written < test:
            f = files[2]
        elif written < test + valid:
            f = files[1]
        elif written < test + valid + train:
            f = files[0]
        else:
            break

        encoded = content.encode("utf-8")
        f.write(encoded)
        f.write(b"\0")
        written += len(encoded) + 1

    for f in files:
        f.close()

    print(f"Wrote {written}/{train + valid + test} for {lang} to {args.output}")


def generate_chinese_markdown(args):
    train, valid, test = map(
        lambda x: int(x * (1024**2)), map(float, args.chinese_markdown_quota.split(","))
    )

    print(
        f"Generating ({train / mb(1)} MB, {valid / mb(1)} MB, {test / mb(1)} MB) for Chinese Markdown"
    )

    chinese_markdown = datasets.load_dataset(
        "rojas-diego/chinese-markdown", split="train", streaming=True
    )

    for split in ["train", "valid", "test"]:
        os.makedirs(f"{args.output}/{split}", exist_ok=True)

    files = [
        open(f"{args.output}/{split}/chinese-markdown.bin", "wb")
        for split in ["train", "valid", "test"]
    ]

    written = 0

    for sample in chinese_markdown:
        content = sample["code"]  # type: ignore

        if written < test:
            f = files[2]
        elif written < test + valid:
            f = files[1]
        elif written < train + valid + test:
            f = files[0]
        else:
            break

        encoded = content.encode("utf-8")
        f.write(encoded)
        f.write(b"\0")
        written += len(encoded) + 1

    for f in files:
        f.close()

    print(f"Wrote Chinese Markdown to {args.output}")


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
        required=True,
        help="The quotas for each language in The Stack in the form {lang}:{train_mb},{valid_mb},{test_mb}",
    )
    parser.add_argument(
        "--chinese-markdown-quota",
        type=str,
        required=True,
        help="The quota for Chinese Markdown data in the form {train_mb},{valid_mb},{test_mb}",
    )
    args = parser.parse_args()

    the_stack_quotas = [quota.split(":") for quota in args.the_stack_quotas]
    the_stack_quotas = [
        (lang, tuple(quota.split(","))) for [lang, quota] in the_stack_quotas
    ]
    the_stack_quotas = {
        lang: (mb(float(train_mb)), mb(float(valid_mb)), mb(float(test_mb)))
        for lang, (train_mb, valid_mb, test_mb) in the_stack_quotas
    }

    threads = []

    for lang, quota in the_stack_quotas.items():
        thread = threading.Thread(target=generate_the_stack, args=(args, lang, quota))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    generate_chinese_markdown(args)
