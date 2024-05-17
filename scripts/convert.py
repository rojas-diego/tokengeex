"""
Converts a TikToken vocabulary to a new TikToken, HF and TokenGeeX vocabulary.
Also supports truncating the vocabulary by pruning low frequency tokens.
"""

import argparse
import glob
from itertools import islice

import numpy as np
import tiktoken
from tiktoken.load import dump_tiktoken_bpe
from tqdm import tqdm


def batch_iterator(iterable, batch_size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        type=str,
        required=True,
        help="The slug name of the vocabulary or the path to the vocabulary file.",
    )
    parser.add_argument(
        "-v",
        type=int,
        help="The size of the new vocabulary. If not provided, the vocabulary will not be truncated.",
    )
    parser.add_argument(
        "-f",
        type=str,
        help="A glob pattern indicating the files on which to count the token frequencies.",
    )
    parser.add_argument(
        "--tokengeex", type=str, help="Output path for the TokenGeeX vocabulary."
    )
    parser.add_argument(
        "--hf", type=str, help="Output path for the Hugging Face vocabulary."
    )
    parser.add_argument(
        "--tiktoken", type=str, help="Output path for the TikToken vocabulary."
    )
    args = parser.parse_args()

    try:
        enc = tiktoken.encoding_for_model(args.i)
        mergeable_ranks = enc._mergeable_ranks
    except Exception:
        raise NotImplementedError(
            "Loading a tiktoken encoding from a file is not supported yet."
        )

    print(f"Name: {enc.name}")
    print(f"Vocab Size: {enc.max_token_value + 1}")

    if args.v and args.f:
        files = glob.glob(args.f)
        files = tqdm(files)
        frequencies = np.zeros(enc.max_token_value + 1, dtype=int)
        for file in files:
            files.set_description(f"Processing {file}")
            with open(file, "rb") as f:
                data = f.read().split(b"\0")
                data = [d.decode("utf-8") for d in data]
                data = tqdm(data, leave=False)

                for batch in batch_iterator(data, 256):
                    batch = enc.encode_ordinary_batch(batch)
                    for ids in batch:
                        for id in ids:
                            frequencies[id] += 1

        print("Sorting frequencies")

        sorted_ids = np.argsort(frequencies)[::-1]

        print("Truncating vocabulary")

        new_vocab_indices = sorted_ids[: args.v]

        print(f"New vocab size: {len(new_vocab_indices)}")

        # First, add all the single-byte tokens
        for i in range(256):
            mergeable_ranks[bytes([i])] = i

        # Then, add the most frequent tokens
        for i, id in enumerate(new_vocab_indices):
            val = enc.decode_single_token_bytes(id)
            if val not in mergeable_ranks:
                mergeable_ranks[val] = i + 256

        mergeable_ranks = {k: v for k, v in mergeable_ranks.items() if v < args.v}

    if args.tiktoken:
        print(f"Saving TikToken vocabulary to {args.tiktoken}")
        dump_tiktoken_bpe(mergeable_ranks, args.tiktoken)

    if args.tokengeex:
        raise NotImplementedError(
            "TokenGeeX vocabulary conversion is not supported yet."
        )

    if args.hf:
        raise NotImplementedError(
            "Hugging Face vocabulary conversion is not supported yet."
        )
