"""
Evaluate a third party tokenizer such as SentencePiece or HuggingFace tokenizers
on the TokenGeeX evaluation set.
"""

import argparse
import glob
import json

import numpy as np
import sentencepiece
import tiktoken
import tokengeex
import tokenizers
import transformers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        required=True,
        type=str,
        help="Tokenization library to use.",
    )
    parser.add_argument(
        "-f",
        required=True,
        type=str,
        help="The path to the SentencePiece model file",
    )
    parser.add_argument(
        "-i",
        required=True,
        type=str,
        help="Glob pattern to the input files to evaluate on",
    )
    parser.add_argument(
        "-o",
        required=True,
        type=str,
        help="The path to the output log file to write the results to",
    )
    args = parser.parse_args()

    encode_fn = None
    vocab_size = None

    if args.l == "tiktoken":
        enc = tiktoken.encoding_for_model(args.f)
        vocab_size = enc.n_vocab

        def encode_tiktoken(text):
            return enc.encode_ordinary(text)

        encode_fn = encode_tiktoken

    elif args.l == "sentencepiece":
        sp = sentencepiece.SentencePieceProcessor(model_file=args.f)  #  type: ignore
        vocab_size = sp.vocab_size()

        def encode_sentencepiece(text):
            return sp.EncodeAsIds(text)

        encode_fn = encode_sentencepiece

    elif args.l == "transformers":
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.f)
        vocab_size = tokenizer.vocab_size

        def encode_transformers(text):
            return tokenizer.encode(text, add_special_tokens=False)

        encode_fn = encode_transformers

    elif args.l == "tokenizers":
        tokenizer = tokenizers.Tokenizer.from_file(args.f)
        vocab_size = tokenizer.get_vocab_size()

        def encode_tokenizers(text):
            return tokenizer.encode(text).ids

        encode_fn = encode_tokenizers

    elif args.l == "tokengeex":
        tokenizer = tokengeex.load(args.f)
        vocab_size = tokenizer.vocab_size()

        def encode_tokengeex(text):
            return tokenizer.encode(text)

        encode_fn = encode_tokengeex

    else:
        raise ValueError(f"Invalid tokenization library: {args.l}")

    num_buckets = 50
    bucket_size = vocab_size // num_buckets
    out = {
        "epoch": 0,
        "split": "test",
        "vocab_size": vocab_size,
        # lang: {num_tokens: int, num_chars: int, chars_per_token: float}
        "compression": {},
        "frequency_buckets": [0] * num_buckets,
        "sample_frequency_buckets": [0] * num_buckets,
    }

    frequency_buckets = np.zeros(vocab_size, dtype=np.int64)
    sample_frequency_buckets = np.zeros(vocab_size, dtype=np.int64)

    for file in glob.glob(args.i):
        filename_base = file.split("/")[-1].split(".")[0]
        samples = open(file).read()
        samples = samples.split("\0")
        num_tokens = 0
        num_chars = 0

        for sample in samples:
            tokens = encode_fn(sample)
            num_tokens += len(tokens)
            num_chars += len(sample)

            for id in tokens:
                frequency_buckets[id] += 1
            for id in set(tokens):
                sample_frequency_buckets[id] += 1

        chars_per_token = round(num_chars / num_tokens, 2)

        out["compression"][filename_base] = {
            "num_tokens": num_tokens,
            "num_chars": num_chars,
            "chars_per_token": chars_per_token,
        }

        print(
            f"{filename_base}, {len(samples)} samples, {num_tokens} tokens, {num_chars} chars, {chars_per_token} chars per token"
        )

    frequency_buckets.sort()
    sample_frequency_buckets.sort()
    frequency_buckets = frequency_buckets[::-1]
    sample_frequency_buckets = sample_frequency_buckets[::-1]

    for i in range(num_buckets):
        out["frequency_buckets"][i] = np.sum(
            frequency_buckets[i * bucket_size : (i + 1) * bucket_size]
        ).item()
        out["sample_frequency_buckets"][i] = np.sum(
            sample_frequency_buckets[i * bucket_size : (i + 1) * bucket_size]
        ).item()

    with open(args.o, "w") as f:
        json.dump(out, f)
        f.write("\n")
