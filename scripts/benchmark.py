import os
import time
from typing import Any, cast

import tiktoken
import tokengeex

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def bytes_to_mb(bytes: int) -> float:
    return round(bytes / 1024 / 1024, 2)


def benchmark_batch(documents: list[str]) -> None:
    num_bytes = sum(map(len, map(str.encode, documents)))

    enc = tiktoken.get_encoding("gpt2")
    enc.encode("warmup")

    start = time.perf_counter_ns()
    enc.encode_ordinary_batch(documents, num_threads=1)
    end = time.perf_counter_ns()
    print(
        f"TikToken     {bytes_to_mb(int(num_bytes / (end-start) * 1e9)):>5} MB/s {round((end - start) / 1e8, 2):>5}s (single thread)"
    )

    import transformers

    hf_enc = cast(Any, transformers).GPT2TokenizerFast.from_pretrained("gpt2")
    hf_enc.model_max_length = 1e30  # silence!
    hf_enc.encode("warmup")
    start = time.perf_counter_ns()
    hf_enc(documents)
    end = time.perf_counter_ns()
    print(
        f"HuggingFace  {bytes_to_mb(int(num_bytes / (end-start) * 1e9)):>5} MB/s {round((end - start) / 1e8, 2):>5}s (single thread)"
    )

    tokenizer = tokengeex.load("./benches/unigram.json")  # type: ignore

    start = time.perf_counter_ns()
    for document in documents:
        tokenizer.encode(document)
    end = time.perf_counter_ns()

    print(
        f"TokenGeex    {bytes_to_mb(int(num_bytes / (end-start) * 1e9)):>5} MB/s {round((end - start) / 1e8, 2):>5}s (single thread)"
    )


samples = open("./benches/data.bin", "rb").read().split(b"\0")

benchmark_batch(list(map(bytes.decode, samples)))
