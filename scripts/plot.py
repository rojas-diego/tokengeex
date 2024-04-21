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
        return json.load(f), args.i.split("/")[-1].split(".")[0]


def plot_cpt(args, data, filename):
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
    plt.gca().set_ylim(top=7)
    offset = 0.35  # This value might need fine-tuning
    plt.xticks(
        ticks=np.arange(len(df["Language"])) + offset,
        labels=df["Language"],  # type: ignore
        rotation=45,
        ha="right",
    )
    plt.title(f"Character per Token Ratio by Language ({filename})")
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
    codegeex_languages = [
        "jsx",
        "javascript",
        "typescript",
        "java",
        "python",
        "html",
        "cpp",
        "c",
    ]
    for subset, name, color in [
        (humanevalx_languages, "HumanEvalX", "green"),
        (codegeex_languages, "CodeGeeX", "blue"),
    ]:
        if all(lang in data.keys() for lang in subset):
            subset_num_tokens = sum(data[lang]["num_tokens"] for lang in subset)
            subset_num_chars = sum(data[lang]["num_chars"] for lang in subset)
            subset_average = subset_num_chars / subset_num_tokens
            # Add a blue dotted line at the subset average
            plt.axhline(
                y=subset_average,
                color=color,
                linestyle="dotted",
                label=f"{name} Average: {subset_average:.2f}",
            )

    # Optionally, add a legend to display the average value
    plt.legend()

    # Save the plot
    if args.cpt:
        plt.savefig(args.cpt, dpi=300)
    else:
        plt.show()


def plot_freq(args, data, filename):
    # {frequency_buckets: [usize], sample_frequency_buckets: [usize]}
    for config in [
        (
            "frequency_buckets",
            "Token Frequency Distribution",
            args.freq,
        ),
    ]:
        key, title, out = config

        pltdata = np.array(data[key], dtype=np.float64)
        pltdata /= pltdata.sum()
        pltdata *= 100

        # Create the figure and axis objects
        fig, ax = plt.subplots(figsize=(10, 6))

        pltdata = pd.DataFrame(
            {"Buckets": range(1, len(pltdata) + 1), "Frequency": pltdata}
        )

        # Plot the data
        sns.barplot(
            x="Buckets",
            y="Frequency",
            data=pltdata,
            ax=ax,
            color="lightblue",
            width=1.0,
        )

        ax.set_yscale("log")
        ax.set_xticklabels([])
        ax.set_ylim(0.0001, 100.0)
        ax.yaxis.set_major_formatter("{x}%")
        ax.set_ylabel("Rate of Occurrence (%)")
        ax.set_title(f"{title} ({filename})")

        # Show the plot
        plt.tight_layout()
        plt.grid(linestyle="dotted")

        # Save the plot
        if out:
            plt.savefig(out, dpi=300)
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

    data, filename = load_data(args)

    plot_cpt(args, data, filename)
    plot_freq(args, data, filename)
