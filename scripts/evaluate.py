"""
This script evaluates a tokenizer on a folder of {lang}.bin files.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Tuple

import numpy as np
from tqdm import tqdm


def get_tokenizer(lib: str, tokenizer: str) -> Tuple[Callable[[str], List[int]], int]:
    encode = None
    vocab_size = 0

    if lib == "transformers":
        from transformers import AutoTokenizer

        t = AutoTokenizer.from_pretrained(tokenizer)

        def transformers_encode(code: str) -> List[int]:
            return t.encode(code, add_special_tokens=False)

        encode = transformers_encode
        vocab_size = t.vocab_size
    elif lib == "tokenizers":
        from tokenizers import Tokenizer

        t = Tokenizer.from_file(tokenizer)

        def tokenizers_encode(code: str) -> List[int]:
            return t.encode(code).ids

        encode = tokenizers_encode
        vocab_size = t.vocab_size()
    elif lib == "tokengeex":
        import tokengeex

        t = tokengeex.load(tokenizer)  # type: ignore

        def tokengeex_encode(code: str) -> List[int]:
            return t.encode(code)

        encode = tokengeex_encode
        vocab_size = t.vocab_size()
    else:
        raise ValueError(f"Unknown library {lib}")

    return encode, vocab_size


def get_samples(path: str) -> Dict[str, List[str]]:
    """Loads all *.bin files in the given path and returns a dictionary of
    language -> list of samples."""

    samples = {}
    for filepath in os.listdir(path):
        if not filepath.endswith(".bin"):
            continue

        with open(f"{path}/{filepath}", "rb") as f:
            lang = filepath.split(".")[0]
            samples[lang] = f.read().decode("utf-8").split("\0")

    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lib",
        type=str,
        help="The library to use (transformers, tokenizers, tokengeex)",
        choices=["transformers", "tokenizers", "tokengeex"],
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="The path to the folder containing the .bin files",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="The path/slug of the tokenizer",
    )

    args = parser.parse_args()

    encode, vocab_size = get_tokenizer(args.lib, args.tokenizer)
    samples = get_samples(args.path)

    metrics = {
        lang: {
            "ntokens": np.array([]),
            "nbytes": np.array([]),
            "nchars": np.array([]),
            "token_frequencies": np.zeros(vocab_size, dtype=np.uint64),
        }
        for lang in samples.keys()
    }

    def process(snippet: str, metrics: Dict[str, np.ndarray]):
        try:
            ids = encode(snippet)
        except Exception as e:
            print(e, file=sys.stderr)
            return

        ntokens = len(ids)
        nbytes = len(snippet.encode("utf-8"))
        nchars = len(snippet)

        metrics["ntokens"] = np.append(metrics["ntokens"], ntokens)
        metrics["nbytes"] = np.append(metrics["nbytes"], nbytes)
        metrics["nchars"] = np.append(metrics["nchars"], nchars)

        for id in ids:
            metrics["token_frequencies"][id] += 1

    pool = ThreadPoolExecutor(8)
    futures = []

    for lang, snippets in samples.items():
        print(f"Processing {lang}...", file=sys.stderr)
        for code in snippets:
            futures.append(pool.submit(process, code, metrics[lang]))

    # This is to ensure all tasks are completed before proceeding
    for future in tqdm(futures, total=len(futures), desc="Waiting"):
        future.result()

    pool.shutdown(wait=True)

    results = {}

    for lang, data in metrics.items():
        ntokens = data["ntokens"].sum()
        nbytes = data["nbytes"].sum()
        nchars = data["nchars"].sum()

        results[lang] = {
            "ndocuments": len(samples[lang]),
            "ntokens": ntokens,
            "nbytes": nbytes,
            "nchars": nchars,
            "token_frequencies": data["token_frequencies"].tolist(),
        }

    print(results)
