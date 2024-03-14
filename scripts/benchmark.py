import os
import sys
import time

import tiktoken

num_threads = int(sys.argv[1]) if len(sys.argv) > 1 else 1

os.environ["TOKENIZERS_PARALLELISM"] = "1"
os.environ["TOKENGEEX_PARALLELISM"] = "1"

os.environ["RAYON_NUM_THREADS"] = str(num_threads)


def bytes_to_mb(bytes: int) -> float:
    return round(bytes / 1024 / 1024, 2)


def benchmark_batch(documents: list[str]) -> None:
    num_bytes = sum(map(len, map(str.encode, documents)))

    enc = tiktoken.get_encoding("gpt2")
    enc.encode("warmup")

    start = time.perf_counter_ns()
    enc.encode_ordinary_batch(documents, num_threads=num_threads)
    end = time.perf_counter_ns()
    print(
        f"TikToken     {bytes_to_mb(int(num_bytes / (end-start) * 1e9)):>5} MB/s {round((end - start) / 1e9, 2):>5}s ({'single thread' if num_threads < 2 else '{} threads'.format(num_threads)})"
    )

    from transformers import GPT2TokenizerFast

    hf_enc = GPT2TokenizerFast.from_pretrained("gpt2")
    hf_enc.model_max_length = 1e30  # silence!
    hf_enc.encode("warmup")
    start = time.perf_counter_ns()
    hf_enc(documents)
    end = time.perf_counter_ns()
    print(
        f"HuggingFace  {bytes_to_mb(int(num_bytes / (end-start) * 1e9)):>5} MB/s {round((end - start) / 1e9, 2):>5}s ({'single thread' if num_threads < 2 else '{} threads'.format(num_threads)})"
    )

    import tokengeex

    tokenizer = tokengeex.load("./data/unigram-65k.json")  # type: ignore

    start = time.perf_counter_ns()
    for doc in documents:
        tokenizer.encode(doc)
    end = time.perf_counter_ns()
    print(
        f"TokenGeex    {bytes_to_mb(int(num_bytes / (end-start) * 1e9)):>5} MB/s {round((end - start) / 1e9, 2):>5}s ({'single thread' if num_threads < 2 else '{} threads'.format(num_threads)})"
    )


samples = open("./data/train.bin", "rb").read().split(b"\0")

samples = samples * 10

benchmark_batch(list(map(bytes.decode, samples)))
