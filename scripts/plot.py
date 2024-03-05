import argparse
import json

import matplotlib.pyplot as plt
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
        "--output", type=str, required=True, help="Path to the output image file"
    )
    args = parser.parse_args()

    # Load the JSON file
    with open(args.json_input, "r") as f:
        json_data = json.load(f)

    # Convert JSON data to pandas DataFrame
    data = pd.DataFrame.from_dict(json_data, orient="index").reset_index()
    data.rename(
        columns={"index": "Language", "chars_per_token": "Characters per Token"},
        inplace=True,
    )

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
    plt.xticks(rotation=45, ha="right")
    plt.title("Character per Token Ratio by Language")
    plt.tight_layout()

    # Calculate the average of the y-values
    average = data["Characters per Token"].mean()

    # Add a red dotted line at the average
    plt.axhline(y=average, color="r", linestyle="--", label=f"Average: {average:.2f}")

    # Optionally, add a legend to display the average value
    plt.legend()

    # Save the plot
    plt.savefig(args.output, dpi=300)
