"""
Generates plots for the characters per token ratio and token frequency
distribution from a training/evaluation log file.
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_data(args):
    with open(args.i, "r") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        epochs = {}

        for event in data:
            event: dict = event
            epoch, split = event["epoch"], event["split"]
            if epoch not in epochs:
                epochs[epoch] = {}
            if split in epochs[epoch]:
                raise ValueError(f"Duplicate split {split} in epoch {epoch}")
            event.pop("epoch")
            event.pop("split")
            epochs[epoch][split] = event

    if args.e == "last":
        epochs_indices = list(map(lambda x: int(x), epochs.keys()))
        epochs_indices.sort()
        epoch = epochs[epochs_indices[-1]]
        print(f"Using epoch {epochs_indices[-1]} of {epochs_indices}")
    else:
        epoch = epochs[int(args.e)]

    return epoch[args.s]


def plot_cpt(args, data):
    # {compression: {lang: {chars_per_token: float, num_tokens: usize, num_chars: usize}}}
    data = data["compression"]

    # Convert JSON data to pandas DataFrame
    df = pd.DataFrame.from_dict(data, orient="index").reset_index()
    df.rename(
        columns={
            "index": "Language",
            "chars_per_token": "Characters per Token",
            "num_chars": "Chars",
        },
        inplace=True,
    )

    # Sort the data by nbytes of each language
    df.sort_values(by="Chars", ascending=False, inplace=True)

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x="Language",
        y="Characters per Token",
        hue="Characters per Token",
        data=df,
        palette="viridis",
        legend=False,
    )
    plt.gca().set_ylim(top=6)
    offset = 0.35  # This value might need fine-tuning
    plt.xticks(
        ticks=np.arange(len(df["Language"])) + offset,
        labels=df["Language"],  # type: ignore
        rotation=45,
        ha="right",
    )
    plt.title("Character per Token Ratio by Language")
    plt.tight_layout()
    plt.xlabel("")

    # Calculate the average of the y-values (by summing all num_tokens and all num_chars and dividing)
    num_tokens = sum(data[lang]["num_tokens"] for lang in data)
    num_chars = sum(data[lang]["num_chars"] for lang in data)
    average = num_chars / num_tokens
    # Add a red dotted line at the average
    plt.axhline(y=average, color="r", linestyle="--", label=f"Average: {average:.2f}")

    # Calculate the average of the HumanEvalX languages
    humanevalx_languages = ["go", "python", "cpp", "java", "javascript"]
    if all(lang in data.keys() for lang in humanevalx_languages):
        humanevalx_num_tokens = sum(
            data[lang]["num_tokens"] for lang in humanevalx_languages
        )
        humanevalx_num_chars = sum(
            data[lang]["num_chars"] for lang in humanevalx_languages
        )
        humanevalx_average = humanevalx_num_chars / humanevalx_num_tokens
        # Add a blue dotted line at the HumanEvalX average
        plt.axhline(
            y=humanevalx_average,
            color="orange",
            linestyle="dotted",
            label=f"HumanEvalX Average: {humanevalx_average:.2f}",
        )

    # Optionally, add a legend to display the average value
    plt.legend()

    # Save the plot
    if args.cpt:
        plt.savefig(args.cpt, dpi=300)
    else:
        plt.show()


def plot_freq(args, data):
    # {frequency_buckets: [usize]}
    data = np.array(data["frequency_buckets"], dtype=np.float64)
    data /= data.sum()
    data *= 100

    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 6))

    data = pd.DataFrame({"Buckets": range(1, len(data) + 1), "Frequency": data})

    # Plot the data
    sns.barplot(
        x="Buckets", y="Frequency", data=data, ax=ax, color="lightblue", width=1.0
    )

    ax.set_yscale("log")
    ax.set_xticklabels([])
    ax.set_ylim(0.0, 100.0)
    ax.yaxis.set_major_formatter("{x}%")
    ax.set_ylabel("Rate of Occurrence (%)")
    ax.set_title("Token Frequency Distribution (Log Scale)")

    # Show the plot
    plt.tight_layout()
    plt.grid(linestyle="dotted")

    # Save the plot
    if args.freq:
        plt.savefig(args.freq, dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the character per token ratio by language."
    )
    parser.add_argument(
        "-i",
        type=str,
        required=True,
        help="Input to the JSONL training/evaluation log file",
    )
    parser.add_argument(
        "-e",
        type=str,
        default="last",
        help="Which epoch to plot",
    )
    parser.add_argument(
        "-s",
        type=str,
        default="test",
        help="Which split to plot",
    )
    parser.add_argument(
        "--cpt",
        type=str,
        help="Path to the output file for characters per token ratio",
    )
    parser.add_argument(
        "--freq",
        type=str,
        help="Path to the output file for token frequency distribution",
    )

    args = parser.parse_args()

    sns.set_theme(
        style="whitegrid",
        palette="pastel",
        font_scale=1.2,
        rc={"font.family": "Times New Roman"},
    )
    plt.rcParams.update({"font.size": 14})

    data = load_data(args)

    plot_cpt(args, data)
    plot_freq(args, data)
