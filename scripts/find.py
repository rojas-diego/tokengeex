"""
Given a specific string, this script searches inside ./hub/data/{split}/*.bin
to find the number of occurrences of the string.
"""

import glob
import sys

path = "./hub/data/{split}/*.bin"


def count_occurrences(string, path):
    print(f"{'file':>40} | {'by occur':>10} | {'by sample':>10} | {'% of samples'}")

    total_occurences = 0
    total_occurences_by_sample = 0
    total_num_samples = 0

    for file in glob.glob(path):
        occurences = 0
        occurences_by_sample = 0
        with open(file, "rb") as f:
            data = f.read()
            data = data.split(b"\0")
            data = [d.decode("utf-8") for d in data]
            num_samples = len(data)
            total_num_samples += num_samples

            for sample in data:
                # Count the number of occurrences of the string in the sample
                count = sample.count(string)
                occurences += count

                if count > 0:
                    occurences_by_sample += 1

        total_occurences += occurences
        total_occurences_by_sample += occurences_by_sample

        occurs_in_pct = (occurences_by_sample / num_samples) * 100

        if occurences > 0:
            print(
                f"{file:>40} | {occurences:>10} | {occurences_by_sample:>10} | {occurs_in_pct:>10.2f}%"
            )

    occurs_in_pct = (total_occurences / total_num_samples) * 100

    print(
        f"{'total':>40} | {total_occurences:>10} | {total_occurences_by_sample:>10} | {occurs_in_pct:>10.2f}%"
    )


if __name__ == "__main__":
    split = sys.argv[1]
    string = sys.argv[2]
    path = path.format(split=split)
    count_occurrences(string, path)
