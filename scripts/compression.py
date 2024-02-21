"""
This script computes the compression ratio for a TokenGeeX tokenizer over a
JSONL dataset. It gives a per-language and overall compression ratio.
"""

import json
import sys

import numpy as np
import tokengeex

assert len(sys.argv) > 2, "Usage: python compression.py <path-to-tokenizer> [lang]"

tokenizer = tokengeex.load(sys.argv[1])

langs = sys.argv[2:]

dataset = {
    lang: [
        json.loads(line)
        for line in open(f"data/test/{lang}.jsonl").read().strip().split("\n")
    ]
    for lang in langs
}


result = {
    lang: {
        "ntokens": np.array([]),
        "nbytes": np.array([]),
        "nchars": np.array([]),
        "lossless": np.array([], dtype=bool),
    }
    for lang in langs
}


for lang, samples in dataset.items():
    for i, sample in enumerate(samples):
        code = sample["code"]  # type: ignore

        try:
            ids = tokenizer.encode(code)
            decoded = tokenizer.decode(ids)
        except:  # noqa: E722
            print("Error tokenizing")
            print("----------------------------------------")
            print(code, end="")
            print("----------------------------------------")
            sys.exit(1)

        lossless = code == decoded
        ntokens = len(ids)
        nbytes = len(code.encode("utf-8"))
        nchars = len(code)

        bytes_per_token = nbytes / ntokens
        chars_per_token = nchars / ntokens

        print(
            f"{lang:<10} | {i+1:>4}/{len(samples):>4} | {nbytes:>6} bytes | {nchars:>6} chars | {ntokens:>6} tokens | {round(bytes_per_token, 2):>4} bytes/token | {('lossless' if lossless else 'lossy'):>10}"
        )

        result[lang]["ntokens"] = np.append(result[lang]["ntokens"], ntokens)
        result[lang]["nbytes"] = np.append(result[lang]["nbytes"], nbytes)
        result[lang]["nchars"] = np.append(result[lang]["nchars"], nchars)
        result[lang]["lossless"] = np.append(result[lang]["lossless"], lossless)


print(
    f"{'lang':<10},{'nbytes':>10},{'nchars':>10},{'ntokens':>10},{'count':>10},{'bpt':>10},{'cpt':>10},{'avg bpt':>10},{'avg cpt':>10},{'lossless':>10}"
)

for lang, data in result.items():
    ntokens = data["ntokens"].sum()
    nbytes = data["nbytes"].sum()
    nchars = data["nchars"].sum()
    count = len(data["ntokens"])
    bpts = data["nbytes"] / data["ntokens"]
    cpts = data["nchars"] / data["ntokens"]
    bpt = bpts.mean()
    cpt = cpts.mean()
    lossless = data["lossless"].mean()

    print(
        f"{lang:<10},{int(nbytes):>10},{int(nchars):>10},{int(ntokens):>10},{int(count):>10},{round(bpt, 2):>10},{round(cpt, 2):>10},{round(nbytes / ntokens, 2):>10},{round(nchars / ntokens, 2):>10},{round(lossless, 2):>10}"
    )
