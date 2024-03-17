import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set the aesthetic style of the plots
sns.set_theme(
    style="whitegrid",
    palette="pastel",
    font_scale=1.2,
    rc={"font.family": "Times New Roman"},
)
plt.rcParams.update({"font.size": 14})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the character per token ratio by language."
    )
    parser.add_argument(
        "--json-input", type=str, required=True, help="Path to the input JSON file"
    )
    parser.add_argument(
        "--npz-input", type=str, required=True, help="Path to the input npz file"
    )
    parser.add_argument(
        "--compression-output",
        type=str,
        required=True,
        help="Path to the output image file",
    )
    parser.add_argument(
        "--frequencies-output",
        type=str,
        required=True,
        help="Path to the output image file",
    )

    args = parser.parse_args()

    # Load the JSON file
    with open(args.json_input, "r") as f:
        json_data = json.load(f)

    # Convert JSON data to pandas DataFrame
    data = pd.DataFrame.from_dict(json_data, orient="index").reset_index()
    data.rename(
        columns={
            "index": "Language",
            "chars_per_token": "Characters per Token",
            "nbytes": "Bytes",
        },
        inplace=True,
    )

    # Sort the data by nbytes of each language
    data.sort_values(by="Bytes", ascending=False, inplace=True)

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x="Language",
        y="Characters per Token",
        hue="Characters per Token",
        data=data,
        palette="viridis",
        legend=False,
    )
    plt.gca().set_ylim(top=6)
    offset = 0.35  # This value might need fine-tuning
    plt.xticks(
        ticks=np.arange(len(data["Language"])) + offset,
        labels=data["Language"],  # type: ignore
        rotation=45,
        ha="right",
    )
    plt.title("Character per Token Ratio by Language")
    plt.tight_layout()
    plt.xlabel("")

    # Calculate the average of the y-values (by summing all ntokens and all nchars and dividing)
    ntokens = sum(json_data[lang]["ntokens"] for lang in json_data)
    nchars = sum(json_data[lang]["nchars"] for lang in json_data)
    average = nchars / ntokens

    # Add a red dotted line at the average
    plt.axhline(y=average, color="r", linestyle="--", label=f"Average: {average:.2f}")

    # Optionally, add a legend to display the average value
    plt.legend()

    # Save the plot
    plt.savefig(args.compression_output, dpi=300)

    # Load the data
    # frequencies = np.array(
    #     [metrics[lang]["token_frequencies"] for lang in results.keys()]
    # )
    # np.savez(args.numpy_output, langs=list(results.keys()), frequencies=frequencies)
    data = np.load(args.npz_input, allow_pickle=True)

    frequencies = data["frequencies"]
    frequencies = frequencies.sum(axis=0)

    # Sum frequencies across all languages
    sum_frequencies = frequencies.sum(axis=0)

    # Determine the number of buckets
    num_buckets = 30

    # Step 2: Calculate the average frequency per bucket
    # Convert the frequencies to a DataFrame
    df = pd.DataFrame(frequencies, columns=["Frequency"])

    # Use qcut to create quantile-based buckets
    df["Bucket"], bins = pd.qcut(
        df["Frequency"],
        q=num_buckets,
        labels=range(num_buckets),
        retbins=True,
        duplicates="drop",
    )

    # Calculate the sum of frequencies for each bucket
    buckets = df.groupby("Bucket")["Frequency"].sum()

    # Divide by the total number of tokens to obtain the percentage
    buckets /= frequencies.sum()
    buckets *= 100

    plot_data = {
        "Bucket": range(num_buckets),
        "Frequency": buckets.reindex(range(num_buckets)).values,
    }

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame(plot_data)

    # Sort the DataFrame in descending order to reverse the curve
    plot_df = plot_df.sort_values(by="Bucket", ascending=False)

    # Melt the DataFrame to have long-form data
    # plot_df_long = pd.melt(plot_df, id_vars=["Bucket"], value_name="Frequency")

    # Set the aesthetic style of the plots
    sns.set_theme(
        style="whitegrid",
        palette="pastel",
        font_scale=1.2,
        rc={"font.family": "Times New Roman"},
    )

    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the data
    sns.barplot(x="Bucket", y="Frequency", data=plot_df, ax=ax, color="lightblue")

    ax.set_yscale("log")

    # Set the y-axis labels as percentages
    ax.yaxis.set_major_formatter("{x}%")

    ax.set_xlabel("Frequency Buckets")
    ax.set_ylabel("Rate of Occurrence (%)")
    ax.set_title("Token Frequency Distribution (Log Scale)")

    # Show the plot
    plt.tight_layout()
    plt.grid(linestyle="dotted")
    plt.savefig(args.frequencies_output, dpi=300)
