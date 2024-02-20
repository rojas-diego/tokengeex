"""
This script computes the compression ratio for a TokenGeeX tokenizer over the
bigcode/the-stack-smol dataset. It gives a per-language and overall compression
ratio.
"""

import random
import sys

import datasets
import numpy as np
import tokengeex

assert len(sys.argv) == 2, "Usage: python compression.py <path-to-tokenizer>"

tokenizer = tokengeex.load(sys.argv[1])

the_stack_smol = datasets.load_dataset("bigcode/the-stack-smol", split="train")
the_stack_smol = the_stack_smol.filter(
    lambda sample: sample["lang"].lower()
    in ["python", "javascript", "java", "go", "c", "c++"]
)
the_stack_smol = the_stack_smol.shuffle(99)

result = {}

for sample in the_stack_smol:
    # Skip 90% of the samples
    if random.random() < 0.9:
        continue

    lang = sample["lang"]
    content = sample["content"]
    repository_name = sample["repository_name"]
    path = sample["path"]

    try:
        ids = tokenizer.encode(tokengeex.capcode.encode(content))
    except:  # noqa: E722
        print(f"Error tokenizing {repository_name}/{path}")
        print("----------------------------------------")
        print(content, end="")
        print("----------------------------------------")
        sys.exit(1)

    if lang not in result:
        result[lang] = {
            "ntokens": np.array([]),
            "nbytes": np.array([]),
            "nchars": np.array([]),
        }

    ntokens = len(ids)
    nbytes = len(content.encode("utf-8"))
    nchars = len(content)

    bytes_per_token = nbytes / ntokens
    chars_per_token = nchars / ntokens

    print(
        f"{lang:>10} | {nbytes:>6} bytes | {ntokens:>6} tokens | {round(bytes_per_token, 2):>4} bytes/token | {repository_name}/{path}"
    )

    # if bytes_per_token < 2:
    #     print("----------------------------------------")
    #     print(content, end="")
    #     print("----------------------------------------")

    result[lang]["ntokens"] = np.append(result[lang]["ntokens"], ntokens)
    result[lang]["nbytes"] = np.append(result[lang]["nbytes"], nbytes)
    result[lang]["nchars"] = np.append(result[lang]["nchars"], nchars)


for lang, data in result.items():
    ntokens = data["ntokens"].sum()
    nbytes = data["nbytes"].sum()
    nchars = data["nchars"].sum()
    count = len(data["ntokens"])
    bpts = data["nbytes"] / data["ntokens"]
    cpts = data["nchars"] / data["ntokens"]
    bpt = bpts.mean()
    cpt = cpts.mean()

    print(
        f"{lang:>10}: {int(nbytes):>10} bytes, {int(nchars):>10} chars, {int(ntokens):>10} tokens, {int(count):>10} samples, {round(bpt, 2):>4} bytes/token, {round(cpt, 2):>4} chars/token"
    )
