"""This script contains code to generate the data descriptive"""

# %% define ROOT
import sys
from pathlib import Path

def load_project_root():
    """
    Makes project root importable in ANY environment:
    - normal scripts
    - scripts in subfolders
    - VS Code interactive window
    - Jupyter notebooks
    """
    try:
        # Try to detect "__file__" — exists for scripts
        current = Path(__file__).resolve()
    except NameError:
        # "__file__" does not exist in notebooks → use cwd
        current = Path.cwd().resolve()

    # Traverse upward to find project_root.py
    for parent in current.parents:
        if (parent / "project_root.py").exists():
            sys.path.append(str(parent))
            return parent

    raise FileNotFoundError("Could not locate project_root.py")

# Execute loader
ROOT = load_project_root()

# Now import project_root module
from project_root import get_project_root

ROOT = get_project_root()
sys.path.append(str(ROOT / "scripts"))
print("PROJECT ROOT:", ROOT)

#%% import libraries
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
# from topic_modelling_bert import *



# Define methods
# %% methods

def plot_articles_by_category_and_source(df, file_name):
    """
    Plot a stacked bar chart showing the number of *articles* per publisher 
    category, broken down by source (gnews vs lexis-nexis).

    Now:
      - Aggregation is done at article-level (unique doc_id).
      - Aesthetics are kept the same.
    """

    # --- Collapse to article-level first ---
    articles = (
        df[["doc_id", "publisher_category", "source"]]
        .drop_duplicates()
    )

    # --- Prepare pivot table: count articles per category & source ---
    pivot = (
        articles.pivot_table(
            index="publisher_category",
            columns="source",
            values="doc_id",
            aggfunc="count",      # number of unique articles
            fill_value=0
        )
    )

    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False)
    pivot_no_total = pivot.drop(columns="total")

    # --- Plot styling (unchanged) ---
    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(14, 8))  # slightly larger

    pivot_no_total.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=["#1f77b4", "#ff7f0e"],
        width=0.85,                       # reduced spacing
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.tick_params(axis='x', labelrotation=45, labelsize=16)
    ax.tick_params(axis='y', labelsize=14)

    # Minimalistic look
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
        )

    # Legend
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

def drop_short_documents(df, text_col="text", min_words=50):
    """
    Removes all rows for any doc_id whose *article-level* word count 
    is below the specified minimum.
    
    Word count is computed from the full article text in `text_col`.
    """

    # Article text per doc_id (unique article)
    article_text = df.groupby("doc_id")[text_col].first()

    # Compute word count of each article
    article_wc = article_text.apply(lambda x: len(str(x).split()))

    # Identify short articles
    short_docs = article_wc[article_wc < min_words].index.tolist()

    print(f"Found {len(short_docs)} short articles (< {min_words} words).")
    if short_docs:
        print(f"Dropping doc_id values: {short_docs}")

    # Drop all rows for short documents
    df_cleaned = df[~df["doc_id"].isin(short_docs)].copy()

    return df_cleaned


def plot_top_publishers_barplot(df, output_path, top_n=10):
    """
    Plot the top-N publishers as a horizontal bar chart using ARTICLE-LEVEL counts
    (unique doc_id per publisher), with category-based colors.

    Also prints a summary table at the end.
    """

    # ============================================================
    # 1. Collapse to article level (one row per article)
    # ============================================================
    articles = (
        df[["doc_id", "publisher", "publisher_category"]]
        .drop_duplicates()
    )

    # ============================================================
    # 2. Compute top-N publishers by number of articles
    # ============================================================
    publisher_counts = (
        articles["publisher"]
        .value_counts()
        .nlargest(top_n)
    )

    top_df = publisher_counts.reset_index()
    top_df.columns = ["publisher", "count"]

    # ============================================================
    # 3. Merge categories
    # ============================================================
    top_df = top_df.merge(
        articles[["publisher", "publisher_category"]].drop_duplicates(),
        on="publisher",
        how="left"
    )

    # Replace missing categories with "other"
    top_df["publisher_category"] = top_df["publisher_category"].fillna("other")

    # ============================================================
    # 4. Color mapping
    # ============================================================
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

    fallback_color = "#999999"

    top_df["color"] = top_df["publisher_category"].apply(
        lambda x: category_palette.get(x, fallback_color)
    )

    # Sort for horizontal bar plot
    top_df = top_df.sort_values("count", ascending=True)

    # ============================================================
    # 5. Plot
    # ============================================================
    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(
        top_df["publisher"],
        top_df["count"],
        color=top_df["color"],
        height=0.65
    )

    # Remove axes clutter
    ax.xaxis.set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")

    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)

    # Value labels
    max_count = top_df["count"].max()
    for i, count in enumerate(top_df["count"]):
        ax.text(
            count + (0.01 * max_count),
            i,
            str(count),
            va="center",
            ha="left",
            fontsize=13
        )

    # Legend (auto-detect included categories)
    present_cats = top_df["publisher_category"].unique()
    legend_handles = [
        plt.matplotlib.patches.Patch(
            color=category_palette.get(cat, fallback_color),
            label=cat
        )
        for cat in present_cats
    ]

    ax.legend(
        handles=legend_handles,
        title="Publisher Category",
        frameon=False,
        loc="lower right",
        fontsize=12,
        title_fontsize=13
    )

    ax.tick_params(axis='y', labelsize=13)
    plt.tight_layout()

    # Save figure
    plt.savefig(
        os.path.join(output_path, "top_publishers_barplot.png"),
        dpi=300,
        transparent=True
    )
    plt.show()

    # ============================================================
    # 6. Print summary table
    # ============================================================
    summary_table = top_df[["publisher", "publisher_category", "count"]]
    print("\n=== Top Publishers (Article-Level Counts) ===\n")
    print(summary_table.to_string(index=False))

    return summary_table


def plot_word_count(df, output_path, sentence_col="sentence_x", x_limit=None):
    """
    Distribution of total word count PER ARTICLE.
    Word count is computed by summing word counts across all sentences for each doc_id.
    """

    # --- Compute sentence-level word counts ---
    df["sentence_word_count"] = df[sentence_col].apply(lambda x: len(str(x).split()))

    # --- Aggregate to article level ---
    article_wc = (
        df.groupby("doc_id")["sentence_word_count"]
        .sum()
    )

    if x_limit:
        article_wc = article_wc[article_wc <= x_limit]

    # --- Stats ---
    stats = {
        "min": int(article_wc.min()),
        "Q1": int(article_wc.quantile(0.25)),
        "median": int(article_wc.median()),
        "Q3": int(article_wc.quantile(0.75)),
        "max": int(article_wc.max()),
    }
    stats_text = "  |  ".join([f"{k}={v}" for k, v in stats.items()])

    # --- Plot ---
    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(16, 4))

    sns.histplot(article_wc, bins=40, stat="density",
                 color="black", fill=False, edgecolor="#cccccc", linewidth=1)

    sns.kdeplot(article_wc, color="#2b6cb0", linewidth=3)
    sns.rugplot(article_wc, color="#2b6cb0", alpha=0.3, height=0.03)

    ax.text(0.5, -0.35, stats_text, transform=ax.transAxes,
            ha="center", fontsize=14, weight="bold")

    ax.set_xlabel("Total Word Count per Article", fontsize=16)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_ticks([])
    ax.grid(False)
    ax.tick_params(axis="x", labelsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "article_word_count.png"), dpi=300, transparent=True)
    plt.show()

def plot_sentence_count(df, output_path, x_limit=None):
    """
    Distribution of number of sentences per article.
    Based on counting unique sentence_id within each doc_id.
    """

    # --- Compute sentence counts ---
    sentence_count = (
        df.groupby("doc_id")["sentence_id"]
        .nunique()
    )

    if x_limit:
        sentence_count = sentence_count[sentence_count <= x_limit]

    # --- Stats ---
    stats = {
        "min": int(sentence_count.min()),
        "Q1": int(sentence_count.quantile(0.25)),
        "median": int(sentence_count.median()),
        "Q3": int(sentence_count.quantile(0.75)),
        "max": int(sentence_count.max()),
    }
    stats_text = "  |  ".join([f"{k}={v}" for k, v in stats.items()])

    # --- Plot ---
    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(16, 4))

    sns.histplot(sentence_count, bins=30, stat="density",
                 color="black", fill=False, edgecolor="#cccccc", linewidth=1)

    sns.kdeplot(sentence_count, color="#2b6cb0", linewidth=3)
    sns.rugplot(sentence_count, height=0.03, color="#2b6cb0", alpha=0.3)

    ax.text(0.5, -0.35, stats_text, transform=ax.transAxes,
            ha="center", fontsize=14, weight="bold")

    ax.set_xlabel("Sentence Count per Article", fontsize=16)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_ticks([])
    ax.grid(False)
    ax.tick_params(axis="x", labelsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "sentence_count_per_article.png"),
                dpi=300, transparent=True)
    plt.show()

def plot_words_per_sentence(df, output_path, sentence_col="sentence_x", x_limit=60):
    """
    Distribution of number of words per SENTENCE.
    Uses the sentence_x (or sentence_y) column.
    """

    # Compute words per sentence
    wps = df[sentence_col].apply(lambda x: len(str(x).split()))

    if x_limit:
        wps = wps[wps <= x_limit]

    # Stats
    stats = {
        "min": int(wps.min()),
        "Q1": int(wps.quantile(0.25)),
        "median": int(wps.median()),
        "Q3": int(wps.quantile(0.75)),
        "max": int(wps.max()),
    }
    stats_text = "  |  ".join([f"{k}={v}" for k, v in stats.items()])

    # Plot
    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(16, 4))

    sns.histplot(wps, bins=40, stat="density",
                 color="black", fill=False, edgecolor="#cccccc", linewidth=1)

    sns.kdeplot(wps, color="#2b6cb0", linewidth=3)
    sns.rugplot(wps, color="#2b6cb0", alpha=0.3, height=0.03)

    ax.text(0.5, -0.35, stats_text, transform=ax.transAxes,
            ha="center", fontsize=14, weight="bold")

    ax.set_xlabel("Words per Sentence", fontsize=16)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_ticks([])
    ax.grid(False)
    ax.tick_params(axis="x", labelsize=14)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "words_per_sentence.png"),
        dpi=300,
        transparent=True
    )
    plt.show()

def drop_long_documents(df, sentence_col="sentence_x", threshold=1000):
    """
    Drops all rows for any doc_id whose total word count exceeds the threshold.
    Word count is computed by summing sentence-level word counts.
    """

    # Compute word count per sentence
    df["sentence_word_count"] = df[sentence_col].apply(lambda x: len(str(x).split()))

    # Aggregate to article-level word count
    doc_word_count = (
        df.groupby("doc_id")["sentence_word_count"]
        .sum()
    )

    # Identify documents exceeding threshold
    long_docs = doc_word_count[doc_word_count > threshold].index.tolist()

    print(f"Removing {len(long_docs)} documents exceeding {threshold} words.")

    # Drop rows with those doc_ids
    df_cleaned = df[~df["doc_id"].isin(long_docs)].copy()

    # Optional: remove helper column
    df_cleaned.drop(columns=["sentence_word_count"], inplace=True, errors="ignore")

    return df_cleaned

# %% main
if __name__ == "__main__":
    ROOT = get_project_root()
    data_path = ROOT / "data/news_corpus/"
    output_path = ROOT / "figures/descriptive/"

    # %%
    # Preprocess the text data
    df["cleaned_text"] = df["translated_text"].apply(preprocess_text, additional_stopwords=["also"])

    df["topics"], topic_model = get_bert_topics(df["cleaned_text"])

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

    #%% Read sentences_svo
    df = pd.read_csv(f"{data_path}/news_corpus_svo.csv.gz", compression="gzip")

    # %%
     df_cleaned = drop_long_documents(df, sentence_col="sentence_x", threshold=1000)
    df_cleaned = drop_short_documents(df_cleaned,min_words=100)
    # %%  Plot the top publisher sources 
    plot_articles_by_category_and_source(df_cleaned, file_name=f"{output_path}/articles_by_category_and_source.png")
    # Generate a table of top publishers
    plot_top_publishers_barplot(df_cleaned, output_path, top_n=10)

    #%% Show the average word count
    print(df_cleaned.groupby('source')['doc_id'].nunique())
    print(f"Total Articles = {df_cleaned.doc_id.nunique()}")
    print(f"Total senteces in the corpus =  {df_cleaned.sentence_global_id.nunique()}")
    print(f"Toal SVOs = {df_cleaned.svo_id.nunique()}")
    plot_word_count(df_cleaned, output_path, x_limit=1000)
    plot_sentence_count(df_cleaned, output_path, x_limit=80)
    plot_words_per_sentence(df_cleaned, output_path=output_path)
    
    # %% drop redundant column
    df_cleaned.drop(columns=['sentences_y'], inplace=True)
    df_cleaned.rename(columns={'sentences_x':'sentences'})

    # %% TODO match ARG0 & ARG1 with actor directory
    
