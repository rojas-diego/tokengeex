from datasets import load_dataset as hf_load_dataset


def mb(x):
    return x * 1024 * 1024


langs = {
    "python": {
        "bytes": mb(150),
    },
    "javascript": {
        "bytes": mb(150),
    },
    "java": {
        "bytes": mb(150),
    },
    "go": {
        "bytes": mb(150),
    },
    "c++": {
        "bytes": mb(150),
    },
    "markdown": {
        "bytes": mb(150),
    },
}


def format_bytes(v):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if v < 1024:
            return f"{int(v)}{unit}"
        v /= 1024
    return f"{int(v)}PB"


total_bytes = sum(langs[lang]["bytes"] for lang in langs.keys())


print(f"Total bytes: {total_bytes}")
print(f"Total bytes: {format_bytes(total_bytes)}")


github_code_clean = hf_load_dataset(
    "codeparrot/github-code",
    split="train",
    trust_remote_code=True,
    streaming=True,
)

with open(f"data/train/code-{format_bytes(total_bytes)}.bin", "wb") as f:
    bytes = {lang: 0 for lang in langs.keys()}

    for i, sample in enumerate(github_code_clean):
        finished = True
        for lang, b in bytes.items():
            if b < langs[lang]["bytes"]:
                finished = False
                break

        if finished:
            break

        if (
            sample["language"].lower() in langs.keys()
            and bytes[sample["language"].lower()]
            < langs[sample["language"].lower()]["bytes"]
            and sample["size"] < 1000
        ):
            bytes[sample["language"].lower()] += sample["size"] + 1
            f.write(sample["code"].encode("utf-8"))
            f.write(b"\0")
