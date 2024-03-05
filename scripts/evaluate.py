"""
This script evaluates a tokenizer on a folder of {lang}.bin files. It outputs a
JSON object with the following structure:

{
    "{lang}": {
        "ndocuments": int,
        "ntokens": int,
        "nbytes": int,
        "nchars": int,
        "bytes_per_token": float,
        "chars_per_token": float,
    }
}

Additionally, it outputs a .npz file with the following format:

| langs | frequencies |
"""

import argparse
import json
import os
import sys
from typing import Callable, Dict, List, Tuple

import numpy as np
import tqdm


def get_tokenizer(
    lib: str, tokenizer: str
) -> Tuple[Callable[[List[str]], List[List[int]]], int]:
    encode = None
    vocab_size = 0

    if lib == "transformers":
        from transformers import AutoTokenizer

        t = AutoTokenizer.from_pretrained(tokenizer)

        def transformers_encode_many(snippets: List[str]) -> List[List[int]]:
            return [t.encode(snippet, add_special_tokens=False) for snippet in snippets]

        encode = transformers_encode_many
        vocab_size = t.vocab_size
    elif lib == "tokenizers":
        from tokenizers import Tokenizer

        t = Tokenizer.from_file(tokenizer)

        def tokenizers_encode_many(snippets: List[str]) -> List[List[int]]:
            return [t.encode(snippet).ids for snippet in snippets]

        encode = tokenizers_encode_many
        vocab_size = t.vocab_size()
    elif lib == "tokengeex":
        import tokengeex

        t = tokengeex.load(tokenizer)  # type: ignore

        def tokengeex_encode_many(snippets: List[str]) -> List[List[int]]:
            return t.encode_many(snippets)

        encode = tokengeex_encode_many
        vocab_size = t.vocab_size()
    elif lib == "tiktoken":
        import tiktoken

        enc = tiktoken.encoding_for_model(tokenizer)

        def tiktoken_encode_many(snippets: List[str]) -> List[List[int]]:
            return enc.encode_ordinary_batch(snippets)

        encode = tiktoken_encode_many
        vocab_size = enc.max_token_value + 1
    else:
        raise ValueError(f"Unknown library {lib}")

    return encode, vocab_size


def get_samples(
    path: str,
) -> Tuple[Dict[str, List[str]], Dict[str, List[int]]]:
    """Loads all *.bin files in the given path and returns a dictionary of
    language -> list of samples."""

    samples = {}
    sample_lengths = {}

    for filepath in os.listdir(path):
        if not filepath.endswith(".bin"):
            continue

        with open(f"{path}/{filepath}", "rb") as f:
            lang = filepath.split(".")[0]
            encoded_samples = f.read().split(b"\0")
            sample_lengths[lang] = [len(s) for s in encoded_samples if s]
            samples[lang] = [s.decode("utf-8") for s in encoded_samples if s]

    return samples, sample_lengths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lib",
        type=str,
        help="The library to use (transformers, tokenizers, tokengeex, tiktoken)",
        choices=["transformers", "tokenizers", "tokengeex", "tiktoken"],
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
    parser.add_argument(
        "--json-output",
        type=str,
        required=True,
        help="The path to the output JSON file",
    )
    parser.add_argument(
        "--numpy-output",
        type=str,
        required=True,
        help="The path to the output Numpy file",
    )

    args = parser.parse_args()

    encode_many, vocab_size = get_tokenizer(args.lib, args.tokenizer)
    samples, sample_lengths = get_samples(args.path)
    total_length = sum(sum(lengths) for lengths in sample_lengths.values())

    metrics = {
        lang: {
            "ntokens": np.zeros(len(samples[lang]), dtype=np.uint64),
            "nbytes": np.zeros(len(samples[lang]), dtype=np.uint64),
            "nchars": np.zeros(len(samples[lang]), dtype=np.uint64),
            "token_frequencies": np.zeros(vocab_size, dtype=np.uint64),
        }
        for lang in samples.keys()
    }

    def process_many(
        offset: int,
        snippets: List[str],
        lengths: List[int],
        metrics: Dict[str, np.ndarray],
    ):
        assert len(snippets) == len(lengths)

        try:
            many_ids = encode_many(snippets)
        except Exception as e:
            print(e, file=sys.stderr)
            return

        for i, (ids, snippet, length) in enumerate(zip(many_ids, snippets, lengths)):
            ntokens = len(ids)
            nchars = len(snippet)

            metrics["ntokens"][offset + i] = ntokens
            metrics["nbytes"][offset + i] = length
            metrics["nchars"][offset + i] = nchars

            for id in ids:
                metrics["token_frequencies"][id] += 1

    batch_size = 4096
    bar = tqdm.tqdm(unit="B", ascii=True, unit_scale=True, total=total_length)
    for lang, snippets in samples.items():
        bar.set_description(f"Processing {lang} ({len(snippets)} snippets)")
        assert len(snippets) == len(sample_lengths[lang])
        for i in range(0, len(snippets), batch_size):
            process_many(
                i,
                snippets[i : i + batch_size],
                sample_lengths[lang][i : i + batch_size],
                metrics[lang],
            )
            bar.update(
                sum(sample_lengths[lang][i : i + batch_size])
                + len(snippets[i : i + batch_size])
            )
    bar.close()

    results = {}

    for lang, data in metrics.items():
        ntokens = data["ntokens"].sum()
        nbytes = data["nbytes"].sum()
        nchars = data["nchars"].sum()

        results[lang] = {
            "ndocuments": len(samples[lang]),
            "ntokens": int(ntokens),
            "nbytes": int(nbytes),
            "nchars": int(nchars),
            "bytes_per_token": float(nbytes / ntokens),
            "chars_per_token": float(nchars / ntokens),
        }

    json_output = json.dumps(
        results,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ": "),
    )

    with open(args.json_output, "w") as f:
        f.write(json_output)

    frequencies = np.array(
        [metrics[lang]["token_frequencies"] for lang in results.keys()]
    )
    np.savez(args.numpy_output, langs=list(results.keys()), frequencies=frequencies)
