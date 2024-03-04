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


def categorize_and_get_proportions(text):
    # Initialize counts for each category
    chinese_count = 0
    english_count = 0
    latin_count = 0
    other_count = 0
    total_count = len(text)

    for char in text:
        cp = ord(char)
        # Categorize character
        if (
            0x4E00 <= cp <= 0x9FFF
            or 0x3400 <= cp <= 0x4DBF
            or 0x20000 <= cp <= 0x2A6DF
            or 0x2A700 <= cp <= 0x2B73F
            or 0x2B740 <= cp <= 0x2B81F
            or 0x2B820 <= cp <= 0x2CEAF
            or 0x2CEB0 <= cp <= 0x2EBEF
            or 0x30000 <= cp <= 0x3134F
        ):
            chinese_count += 1
        elif "a" <= char.lower() <= "z":
            english_count += 1
        elif 0x0000 <= cp <= 0x00FF:
            latin_count += 1
        else:
            other_count += 1

    # Calculate proportions
    chinese_prop = chinese_count / total_count
    english_prop = english_count / total_count
    latin_prop = latin_count / total_count
    other_prop = other_count / total_count

    return chinese_prop, english_prop, latin_prop, other_prop


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
        "--en-issues-quota",
        type=str,
        required=True,
        help="The quota for English issues data in the form {train_mb},{valid_mb},{test_mb}",
    )
    parser.add_argument(
        "--zh-issues-quota",
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

    print(f"en issues  {args.en_issues_quota:<5}")
    print(f"zh issues  {args.zh_issues_quota:<5}")

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

    is_done = False

    zh_written = 0
    en_written = 0

    for split in ["train", "valid", "test"]:
        os.makedirs(f"{args.output}/{split}", exist_ok=True)

    en_train, en_valid, en_test = map(
        lambda x: x * (1024**2), map(int, args.en_issues_quota.split(","))
    )
    zh_train, zh_valid, zh_test = map(
        lambda x: x * (1024**2), map(int, args.zh_issues_quota.split(","))
    )

    zh_files = [
        open(f"{args.output}/{split}/zh-issues.bin", "wb")
        for split in ["train", "valid", "test"]
    ]

    en_files = [
        open(f"{args.output}/{split}/en-issues.bin", "wb")
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

                chinese_prop, english_prop, latin_prop, other_prop = (
                    categorize_and_get_proportions(text)
                )

                if chinese_prop > 0.05:
                    if zh_written < zh_train:
                        f = zh_files[0]
                    elif zh_written < zh_train + zh_valid:
                        f = zh_files[1]
                    elif zh_written < zh_train + zh_valid + zh_test:
                        f = zh_files[2]
                    else:
                        if en_written >= en_train + en_valid + en_test:
                            is_done = True
                        break

                    print(
                        f"Wrote chinese ({zh_written:>10}/{zh_train+zh_valid+zh_test:>10} bytes)"
                    )
                    f.write(text.encode("utf-8"))
                    f.write(b"\0")
                    zh_written += len(text)

                elif english_prop > 0.5:
                    if en_written < en_train:
                        f = en_files[0]
                    elif en_written < en_train + en_valid:
                        f = en_files[1]
                    elif en_written < en_train + en_valid + en_test:
                        f = en_files[2]
                    else:
                        if zh_written >= zh_train + zh_valid + zh_test:
                            is_done = True
                        break

                    print(
                        f"Wrote english ({en_written:>10}/{en_train+en_valid+en_test:>10} bytes)"
                    )
                    f.write(text.encode("utf-8"))
                    f.write(b"\0")
                    en_written += len(text)

    for f in zh_files + en_files:
        f.close()

    for split in ["train", "valid", "test"]:
        with open(f"{args.output}/{split}/zh-issues.bin", "rb") as f:
            s3.upload_fileobj(
                f,
                args.bucket,
                f"{args.bucket_prefix}/{split}/zh-issues.bin",
                ExtraArgs={"ACL": "public-read"},
            )

        with open(f"{args.output}/{split}/en-issues.bin", "rb") as f:
            s3.upload_fileobj(
                f,
                args.bucket,
                f"{args.bucket_prefix}/{split}/en-issues.bin",
                ExtraArgs={"ACL": "public-read"},
            )

    print(f"Uploaded issues data to {args.bucket}/{args.bucket_prefix}")
