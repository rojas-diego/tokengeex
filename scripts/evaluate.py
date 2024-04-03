"""
Evaluate a third party tokenizer such as SentencePiece or HuggingFace tokenizers
on the TokenGeeX evaluation set.
"""

import argparse
import glob
import json

import sentencepiece as spm
import tiktoken
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
        sp = spm.SentencePieceProcessor(model_file=args.f)  #  type: ignore
        vocab_size = sp.vocab_size()

        def encode_sentencepiece(text):
            return sp.EncodeAsIds(text)

        encode_fn = encode_sentencepiece

    elif args.l == "huggingface":
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.f)
        vocab_size = tokenizer.vocab_size

        def encode_huggingface(text):
            return tokenizer.encode(text, add_special_tokens=False)

        encode_fn = encode_huggingface

    elif args.l == "tokenizers":
        tokenizer = tokenizers.Tokenizer.from_file(args.f)
        vocab_size = tokenizer.get_vocab_size()

        def encode_tokenizers(text):
            return tokenizer.encode(text).ids

        encode_fn = encode_tokenizers

    else:
        raise ValueError(f"Invalid tokenization library: {args.l}")

    out = {
        "epoch": 0,
        "split": "test",
        "vocab_size": vocab_size,
        # lang: {num_tokens: int, num_chars: int, chars_per_token: float}
        "compression": {},
        "frequency_buckets": [0 for i in range(25)],
    }

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

        chars_per_token = round(num_chars / num_tokens, 2)

        out["compression"][filename_base] = {
            "num_tokens": num_tokens,
            "num_chars": num_chars,
            "chars_per_token": chars_per_token,
        }

        print(
            f"{filename_base}, {len(samples)} samples, {num_tokens} tokens, {num_chars} chars, {chars_per_token} chars per token"
        )

    with open(args.o, "w") as f:
        json.dump(out, f)
        f.write("\n")
