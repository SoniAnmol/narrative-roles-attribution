"""This script contains code to generate the data descriptive"""

# %%
# import libraries
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from appendix.topic_modelling_bert import *

# Define methods
# %%

def plot_articles_by_category_and_source(df, file_name):
    """
    Plot a stacked bar chart showing the number of articles per publisher 
    category, broken down by source (gnews vs lexis-nexis).

    Improvements:
        - Larger font sizes for readability
        - Reduced spacing between bars
        - Minimalistic storytelling style (no grid, no y-axis)
        - Total counts displayed above bars
    """

    # --- Prepare pivot table ---
    pivot = (
        df.pivot_table(
            index="publisher_category",
            columns="source",
            values="title",
            aggfunc="count",
            fill_value=0
        )
    )

    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False)
    pivot_no_total = pivot.drop(columns="total")

    # --- Plot styling ---
    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(14, 8))  # slightly larger

    # Plot with reduced bar spacing (larger width)
    pivot_no_total.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=["#1f77b4", "#ff7f0e"],
        width=0.85,                       # << reduced spacing
    )

    # --- Title & labels ---
    # ax.set_title(
    #     "Article Volume by Publisher Category and Source",
    #     fontsize=22,
    #     weight="bold",
    #     pad=20
    # )

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.tick_params(axis='x', labelrotation=45, labelsize=16)
    ax.tick_params(axis='y', labelsize=14)

    # Remove chartjunk
    ax.get_yaxis().set_visible(False)
    ax.grid(False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    # --- Add total value labels ---
    for idx, total in enumerate(pivot["total"]):
        ax.text(
            idx,
            total + (total * 0.015),
            str(total),
            ha="center",
            va="bottom",
            fontsize=14,
            # weight="bold"
        )

    # --- Legend ---


    ax.legend(
        title="Source",
        title_fontsize=14,
        fontsize=13,
        frameon=False,
        loc="upper right"
    )

    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()

def plot_top_publishers_barplot(df, output_path, column="publisher", top_n=10):
    """
    Plot the top-N publishers as a horizontal bar chart with category-based colors.
    Robust to unseen or missing categories.
    """

    # Compute top publishers
    top_publishers = df[column].value_counts().nlargest(top_n)

    top_df = top_publishers.reset_index()
    top_df.columns = ["publisher", "count"]

    # Merge categories
    top_df = top_df.merge(
        df[["publisher", "publisher_category"]].drop_duplicates(),
        on="publisher",
        how="left"
    )

    # Replace NaN categories with "other"
    top_df["publisher_category"] = top_df["publisher_category"].fillna("other")

    # Base palette (extendable)
    category_palette = {
        "national": "#1f77b4",
        "local": "#ff7f0e",
        "regional": "#2ca02c",
        "digital": "#9467bd",
        "institutional": "#8c564b",
        "advocacy": "#e377c2",
        "broadcast": "#7f7f7f",
        "international": "#17becf",
        "other": "#bcbd22",
    }

    # Fallback color for unknown categories
    fallback_color = "#999999"

    # ---- SAFE color mapping (no KeyErrors) ----
    top_df["color"] = top_df["publisher_category"].apply(
        lambda x: category_palette.get(x, fallback_color)
    )

    # Sort in descending order for barh
    top_df = top_df.sort_values("count", ascending=True)

    # ---- Plot ----
    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(
        top_df["publisher"],
        top_df["count"],
        color=top_df["color"],
        height=0.65
    )

    # Title
    # ax.set_title(
    #     f"Top {top_n} Publishers",
    #     fontsize=20,
    #     weight="bold",
    #     pad=15
    # )

    # Turn off x-axis
    ax.xaxis.set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Remove spines
    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)

    # Data labels
    max_count = top_df["count"].max()
    for i, count in enumerate(top_df["count"]):
        ax.text(
            count + (0.01 * max_count),
            i,
            str(count),
            va="center",
            ha="left",
            fontsize=14,
            # weight="bold"
        )

    # ------------- LEGEND (Robust & Dynamic) ----------------
    # Only categories actually present
    present_categories = top_df["publisher_category"].unique()

    legend_handles = [
        plt.matplotlib.patches.Patch(
            color=category_palette.get(cat, fallback_color),
            label=cat
        )
        for cat in present_categories
    ]

    ax.legend(
        handles=legend_handles,
        title="Publisher Category",
        frameon=False,
        loc="lower right",
        fontsize=12,
        title_fontsize=13
    )

    # Y-axis labels readable
    ax.tick_params(axis='y', labelsize=13)

    plt.tight_layout()

    # Save
    plt.savefig(
        os.path.join(output_path, "top_publishers_barplot.png"),
        dpi=300,
        transparent=True
    )

    plt.show()

def plot_word_count(df, output_path, text_column="text", x_limit=None):
    """
    Clean storytelling distribution plot:
        - thin outline histogram (no fill)
        - smooth KDE curve
        - subtle rug for distribution points
        - key stats annotated on top
        - clean, minimal, transparent style
    """

    # Compute word count
    wc = df[text_column].apply(lambda x: len(x.split()))
    
    if x_limit:
        wc = wc[wc <= x_limit]

    # Stats
    stats = {
        "min": wc.min(),
        "Q1": int(wc.quantile(0.25)),
        "median": int(wc.median()),
        "Q3": int(wc.quantile(0.75)),
        "max": wc.max(),
    }
    stats_text = "  |  ".join([f"{k}={v}" for k, v in stats.items()])

    # --- Plot ---
    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(16, 4))

    # Light outline histogram (NO bars)
    sns.histplot(
        wc,
        bins=40,
        stat="density",
        color="black",
        fill=False,
        edgecolor="#cccccc",
        linewidth=1,
        ax=ax
    )

    # KDE
    sns.kdeplot(
        wc,
        color="#2b6cb0",
        linewidth=3,
        ax=ax
    )

    # Rug (subtle)
    sns.rugplot(
        wc,
        ax=ax,
        color="#2b6cb0",
        alpha=0.3,
        height=0.03
    )

    # Title
    # ax.set_title(
    #     "Distribution of Article Word Count",
    #     fontsize=22,
    #     weight="bold",
    #     pad=25
    # )

    # Stats annotation
    ax.text(
        0.5,
        -0.35,
        stats_text,
        transform=ax.transAxes,
        ha="center",
        fontsize=14,
        weight="bold"
    )

    # Labels
    ax.set_xlabel("Word Count", fontsize=16)
    # ax.set_ylabel("Density", fontsize=16)

    # CLEAN THE LOOK
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelsize=14)
    ax.grid(False)
    ax.get_yaxis().set_ticks([])   # no y axis clutter

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "word_count.png"),
        dpi=300,
        transparent=True
    )
    plt.show()

def plot_sentence_count(df, output_path, text_column="text", x_limit=None):
    """
    Clean storytelling distribution plot for sentence counts:
        - thin outline histogram
        - smooth KDE curve
        - subtle rug plot
        - key stats annotated above the title
        - minimal FT-style aesthetic
    """

    # --- Sentence Count Extraction ---
    def count_sentences(text):
        # Count ., ?, ! but avoid counting abbreviations like "e.g." or "U.S."
        # Simple heuristic: sentence-ending punctuation followed by space or end of string.
        return len(re.findall(r"[.!?](\s|$)", text))

    sentence_count = df[text_column].apply(count_sentences)

    if x_limit:
        sentence_count = sentence_count[sentence_count <= x_limit]

    # --- Stats ---
    stats = {
        "min": sentence_count.min(),
        "Q1": int(sentence_count.quantile(0.25)),
        "median": int(sentence_count.median()),
        "Q3": int(sentence_count.quantile(0.75)),
        "max": sentence_count.max(),
    }
    stats_text = "  |  ".join([f"{k}={v}" for k, v in stats.items()])

    # --- Plot ---
    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(16, 4))

    # Light outline histogram
    sns.histplot(
        sentence_count,
        bins=30,
        stat="density",
        color="black",
        fill=False,
        edgecolor="#cccccc",
        linewidth=1,
        ax=ax
    )

    # KDE curve (hero)
    sns.kdeplot(
        sentence_count,
        color="#2b6cb0",
        linewidth=3,
        ax=ax
    )

    # Subtle rug markers
    sns.rugplot(
        sentence_count,
        height=0.03,
        color="#2b6cb0",
        alpha=0.3,
        ax=ax
    )

    # Title
    # ax.set_title(
    #     "Distribution of Sentence Count",
    #     fontsize=22,
    #     weight="bold",
    #     pad=25
    # )

    # Stats annotation below title
    ax.text(
        0.5,
        -0.35,
        stats_text,
        transform=ax.transAxes,
        ha="center",
        fontsize=14,
        weight="bold"
    )

    # Labels
    ax.set_xlabel("Sentence Count", fontsize=16)
    # ax.set_ylabel("Density", fontsize=16)

    # Clean aesthetics
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ax.get_yaxis().set_ticks([])
    ax.tick_params(axis="x", labelsize=14)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "sentence_count_distribution.png"),
        dpi=300,
        transparent=True
    )
    plt.show()



# %%
# main
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent 
    data_path = ROOT / "data/news_corpus/"
    output_path = ROOT / "figures/descriptive/"

    # read data
    # %%
    df = pd.read_csv(f"{data_path}/news_corpus.csv.gz", compression="gzip")

    # %%
    # Generate chart to show the source of articles
    plot_articles_by_category_and_source(df, file_name=f"{output_path}/articles_by_category_and_source.png")
    # %%
    # Plot top publishers
    plot_top_publishers_barplot(df, output_path, column="publisher", top_n=10)

    # %%
    # Show the average word count
    plot_word_count(df, output_path, text_column="text", x_limit=1000)
    # %%
    # show the average sentence count
    plot_sentence_count(df, output_path, text_column="text", x_limit=80)
    # %%
    # Preprocess the text data
    df["cleaned_text"] = df["translated_text"].apply(preprocess_text, additional_stopwords=["also"])

    df["topics"], topic_model = get_bert_topics(df["cleaned_text"])

    # %%
    top_n = 10  # Change this to your desired number of topics
    # Group by date and topic to count the occurrences
    df = df[df["date"].notna()]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    topic_over_time = df.groupby([df["date"].dt.to_period("M"), "topics"]).size().unstack().fillna(0)

    # --------------------------------- Plotting --------------------------------- #
    # Get the total occurrence of each topic across all time periods
    topic_totals = topic_over_time.sum(axis=0).sort_values(ascending=False)

    # Select only the top_n topics based on total occurrence
    top_topics = topic_totals.head(top_n).index
    topic_over_time_top = topic_over_time[top_topics]

    # Get topic information with top words
    topic_info = topic_model.get_topic_info()  # Assume this contains topic IDs and top words
    top_words = topic_info.set_index("Topic").loc[top_topics, "Name"]  # Extract top words for each topic

    cleaned_top_words = top_words.apply(lambda x: x.split("_", 1)[-1])

    # --------------------------------- Plotting --------------------------------- #
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot only the top_n topics
    topic_over_time_top.plot(ax=ax, kind="line", marker="o")

    # Update legend to include topic names with top words
    legend_labels = [f"Topic Words: {cleaned_top_words.loc[topic_id]}" for topic_id in top_topics]
    ax.legend(legend_labels, title="Top words in the topic", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Wave 1: May 2–3, 2023
    ax.axvspan(pd.Timestamp("2023-05-02"), pd.Timestamp("2023-05-03"), color="red", alpha=0.9)

    ax.text(
        pd.Timestamp("2023-05-15"),
        topic_over_time_top.values.max() + 10,
        "Floods",
        color="red",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )

    plt.title("Top Topics Emergence Over Time")
    plt.xlabel("Time")
    plt.ylabel("Number of Articles")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "top_topics_over_time.png"), dpi=300, transparent=True)
    plt.show()
# %%
