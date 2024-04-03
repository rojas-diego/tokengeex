"""
This script trains a baseline tokenizer model using the HuggingFace or
SentencePiece library. It is used to compare the performance of TokenGeeX with
other tokenization libraries.
"""

import argparse
import glob
from typing import Iterable

import sentencepiece as spm
from tokenizers import Tokenizer, models, trainers
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelEncoder


def load_samples(input: str, proportion: float) -> Iterable[str]:
    files = glob.glob(f"{input}/*.bin")
    print(f"Found {len(files)} .bin files in {input}")
    # Each .bin file is a 0x00 separated list of UTF-8 samples.
    for file in files:
        with open(file, "rb") as f:
            samples = f.read().decode("utf-8").split("\0")
            print(f"Loaded {len(samples)} samples from {file}")
            yield from samples[: int(len(samples) * proportion)]


def train_huggingface(samples: Iterable[str], output: str, vocab_size: int):
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        max_token_length=32,
        show_progress=True,
    )  # type: ignore

    tokenizer = Tokenizer(
        model=models.BPE(
            byte_fallback=True,
            fuse_unk=False,
            unk_token=None,
        )
    )
    tokenizer.pre_tokenizer = ByteLevelEncoder(
        add_prefix_space=False,
    )  # type: ignore
    tokenizer.decoder = ByteLevelDecoder()  # type: ignore

    tokenizer.train_from_iterator(samples, trainer=trainer)

    tokenizer.save(output)


def train_sentencepiece(samples: Iterable[str], output: str, vocab_size: int):
    spm.SentencePieceTrainer.Train(
        sentence_iterator=samples,
        vocab_size=vocab_size,
        model_prefix=output,
        model_type="bpe",
        input_sentence_size=10000000,
        shuffle_input_sentence=True,
        num_threads=100,
        split_digits=True,
        byte_fallback=True,
        character_coverage=0.999,
        max_sentencepiece_length=32,
        add_dummy_prefix=False,
        allow_whitespace_only_pieces=True,
        split_by_whitespace=False,
        remove_extra_whitespaces=False,
        normalization_rule_name="identity",
        train_extremely_large_corpus=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer model.")
    parser.add_argument(
        "-l",
        required=True,
        type=str,
        choices=["huggingface", "sentencepiece"],
        help="The library to use for training the tokenizer.",
    )
    parser.add_argument(
        "-i",
        required=True,
        type=str,
        help="The directory containing the training data.",
    )
    parser.add_argument(
        "-o",
        required=True,
        type=str,
        help="The file to save the trained model to.",
    )
    parser.add_argument(
        "-v",
        required=True,
        type=int,
        help="The size of the vocabulary to use.",
    )
    parser.add_argument(
        "-p",
        default=1.0,
        type=float,
        help="Proportion of samples to use for training (between 0 and 1).",
    )

    args = parser.parse_args()

    lib, input, output, vocab_size, proportion = args.l, args.i, args.o, args.v, args.s

    print(
        f"Training {lib} BPE model with {vocab_size} vocabulary size from {input}. Writing to {output}."
    )

    samples = load_samples(input, proportion)

    if lib == "huggingface":
        train_huggingface(samples, output, vocab_size)
    elif lib == "sentencepiece":
        train_sentencepiece(samples, output, vocab_size)
    else:
        raise ValueError(f"Unknown library: {lib}")
