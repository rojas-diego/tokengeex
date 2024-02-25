"""
This script trains a tokenizer model using the HuggingFace library. It is used
to compare the performance of TokenGeeX with the HuggingFace library.
"""

import argparse

from tokenizers import Tokenizer, models, trainers
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelEncoder
from tokenizers.pre_tokenizers import Metaspace as MetaspaceEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a tokenizer model using the HuggingFace library."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["bpe", "unigram"],
        help="The kind of model to train.",
    )
    parser.add_argument(
        "--file", type=str, help="The file containing the training data."
    )
    parser.add_argument(
        "--output", type=str, help="The file to save the trained model to."
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=16394,
        help="The size of the vocabulary to use.",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="bytelevel",
        choices=["metaspace", "bytelevel"],
        help="The encoder/decoder to use.",
    )
    args = parser.parse_args()

    print(f"Training {args.model} model with {args.vocab_size} vocabulary size")

    model = (
        models.BPE(byte_fallback=True, fuse_unk=False)
        if args.model == "bpe"
        else models.Unigram(vocab=[("<UNK>", 0)], unk_id=0, byte_fallback=True)
    )
    trainer = (
        trainers.BpeTrainer(
            vocab_size=args.vocab_size, limit_alphabet=256, max_token_length=24
        )  # type: ignore
        if args.model == "bpe"
        else trainers.UnigramTrainer(
            vocab_size=args.vocab_size,
            unk_token="<UNK>",
        )
    )

    tokenizer = Tokenizer(model)

    if args.codec == "bytelevel":
        tokenizer.pre_tokenizer = ByteLevelEncoder()  # type: ignore
        tokenizer.decoder = ByteLevelDecoder()  # type: ignore
    else:
        tokenizer.pre_tokenizer = MetaspaceEncoder()  # type: ignore
        tokenizer.decoder = MetaspaceDecoder()  # type: ignore

    file = open(args.file, "rb")
    samples = file.read().decode("utf-8").split("\0")

    print(f"Training tokenizer with {len(samples)} samples")

    tokenizer.train_from_iterator(iter(samples), trainer=trainer)

    tokenizer.save(args.output)
