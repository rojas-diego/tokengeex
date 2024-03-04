"""
Utility script to construct the pre-training dataset form The Stack v1.2
deduplicated dataset based on per-language quotas.
It uploads the dataset to CloudFlare R2 as
{bucket}/{bucket_prefix}/{split}/{lang}.bin.
"""

import argparse
import os

import boto3
import datasets


def mb(size: int) -> int:
    return size * (1024**2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="The directory in which to write the temporary files",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        required=True,
        help="The name of the bucket to upload the dataset to",
    )
    parser.add_argument(
        "--bucket-endpoint",
        type=str,
        required=True,
        help="The endpoint of the bucket to upload the dataset to",
    )
    parser.add_argument(
        "--bucket-access-key-id",
        type=str,
        required=True,
        help="The access key ID for the bucket",
    )
    parser.add_argument(
        "--bucket-secret-access-key",
        type=str,
        required=True,
        help="The secret access key for the bucket",
    )
    parser.add_argument(
        "--bucket-region",
        type=str,
        required=True,
        help="The region of the bucket to upload the dataset to",
    )
    parser.add_argument(
        "--bucket-prefix",
        type=str,
        required=True,
        help="Path in the bucket in which to place the .bin files",
    )
    parser.add_argument(
        "--pl-quotas",
        type=str,
        nargs="+",
        required=True,
        help="The quotas for each language in the form {lang}:{train_mb},{valid_mb},{test_mb}",
    )
    parser.add_argument(
        "--issues-quota",
        type=str,
        required=True,
        help="The quota for Chinese issues data in the form {train_mb},{valid_mb},{test_mb}",
    )

    args = parser.parse_args()

    pl_quotas = [quota.split(":") for quota in args.pl_quotas]
    pl_quotas = [(lang, tuple(quota.split(","))) for [lang, quota] in pl_quotas]
    pl_quotas = {
        lang: (mb(int(train_mb)), mb(int(valid_mb)), mb(int(test_mb)))
        for lang, (train_mb, valid_mb, test_mb) in pl_quotas
    }

    print("Language   Train Valid Test")
    for lang, (train, valid, test) in pl_quotas.items():
        print(
            f"{lang:<10} {int(train / mb(1)):<5} {int(valid / mb(1)):<5} {int(test / mb(1)):<5}"
        )

    pl_data = {
        lang: datasets.load_dataset(
            "bigcode/the-stack-dedup",
            data_dir=f"data/{lang}",
            split="train",
            streaming=True,
        )
        for lang in pl_quotas.keys()
    }

    issues = datasets.load_dataset(
        "bigcode/the-stack-github-issues", split="train", streaming=True
    )

    s3 = boto3.client(
        service_name="s3",
        endpoint_url=args.bucket_endpoint,
        aws_access_key_id=args.bucket_access_key_id,
        aws_secret_access_key=args.bucket_secret_access_key,
        region_name=args.bucket_region,
    )

    for lang, (train, valid, test) in pl_quotas.items():
        if all(
            os.path.exists(f"{args.output}/{split}/{lang}.bin")
            for split in ["train", "valid", "test"]
        ):
            print(f"Skipping {lang} as it already exists")
            continue

        for split in ["train", "valid", "test"]:
            os.makedirs(f"{args.output}/{split}", exist_ok=True)

        written = 0
        files = [
            open(f"{args.output}/{split}/{lang}.bin", "wb")
            for split in ["train", "valid", "test"]
        ]

        for sample in pl_data[lang]:
            (
                content,
                size,
                ext,
                avg_line_length,
                max_line_length,
                alphanum_fraction,
                path,
            ) = (
                sample["content"],  # type: ignore
                sample["size"],  # type: ignore
                sample["ext"],  # type: ignore
                sample["avg_line_length"],  # type: ignore
                sample["max_line_length"],  # type: ignore
                sample["alphanum_fraction"],  # type: ignore
                sample["max_stars_repo_path"],  # type: ignore
            )

            if avg_line_length > 100:
                continue

            if alphanum_fraction < 0.5:
                continue

            if max_line_length > 1000:
                continue

            if written < train:
                f = files[0]
            elif written < train + valid:
                f = files[1]
            elif written < train + valid + test:
                f = files[2]
            else:
                break

            f.write(content.encode("utf-8"))
            f.write(b"\0")
            written += size

        for f in files:
            f.close()

        for split in ["train", "valid", "test"]:
            with open(f"{args.output}/{split}/{lang}.bin", "rb") as f:
                s3.upload_fileobj(
                    f,
                    args.bucket,
                    f"{args.bucket_prefix}/{split}/{lang}.bin",
                    ExtraArgs={"ACL": "public-read"},
                )

        print(f"Uploaded {lang} data to {args.bucket}/{args.bucket_prefix}")

    is_done = False
    written = 0

    for split in ["train", "valid", "test"]:
        os.makedirs(f"{args.output}/{split}", exist_ok=True)

    train, valid, test = map(
        lambda x: x * (1024**2), map(int, args.issues_quota.split(","))
    )

    files = [
        open(f"{args.output}/{split}/issues.bin", "wb")
        for split in ["train", "valid", "test"]
    ]

    for sample in issues:
        if is_done:
            break

        events = sample["events"]  # type: ignore

        for event in events:
            if "text" in event:
                text = event["text"]

                if len(text) == 0:
                    continue

                if written < train:
                    f = files[0]
                elif written < train + valid:
                    f = files[1]
                elif written < train + valid + test:
                    f = files[2]
                else:
                    is_done = True
                    break

                f.write(text.encode("utf-8"))
                f.write(b"\0")
                written += len(text)

    for f in files:
        f.close()

    for split in ["train", "valid", "test"]:
        with open(f"{args.output}/{split}/issues.bin", "rb") as f:
            s3.upload_fileobj(
                f,
                args.bucket,
                f"{args.bucket_prefix}/{split}/issues.bin",
                ExtraArgs={"ACL": "public-read"},
            )

    print(f"Uploaded issues data to {args.bucket}/{args.bucket_prefix}")
