"""This script contains code to generate the data descriptive"""

# %%
# import libraries
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import os
from appendix.topic_modelling_bert import *


# %%
# Define methods
def plot_pie_chart(df, output_path, column="source"):
    """Generates a pie chart of the article sources and saves it as an image."""
    source_counts = df[column].value_counts()
    source_labels = source_counts.index
    source_sizes = source_counts.values
    source_percentages = 100 * source_sizes / source_sizes.sum()

    plt.figure(figsize=(8, 6))
    plt.pie(
        source_sizes,
        labels=[
            f"{label} ({count}, {percentage:.1f}%)"
            for label, count, percentage in zip(source_labels, source_sizes, source_percentages)
        ],
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops=dict(edgecolor="k"),
    )
    plt.title(f"Source of {source_sizes.sum()} Articles")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "source_pie_chart.png"), dpi=300, transparent=True)
    plt.show()


# %%
def plot_top_publishers_barplot(df, output_path, column="publishers", top_n=10):
    """Generates a bar plot of the top N publishers and saves it as an image."""
    top_publishers = df[column].value_counts().nlargest(top_n)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=top_publishers.values, y=top_publishers.index)
    plt.title(f"Top {top_n} Publishers")
    plt.xlabel("Publishers")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "top_publishers_barplot.png"), dpi=300, transparent=True)
    # plt.close()
    plt.show()


# %%
def plot_word_count(df, output_path, text_column="text", x_limit=None):
    """Generates a vertical violin plot showing the distribution of word count with whiskers and saves it as an image."""
    df["word_count"] = df[text_column].apply(lambda x: len(x.split()))

    plt.figure(figsize=(8, 6))
    sns.violinplot(y=df["word_count"], inner="box", orient="v")
    plt.title("Distribution of Word Count")
    plt.ylabel("Word Count")

    # Apply y-axis limit if specified
    if x_limit:
        plt.ylim(0, x_limit)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "word_count_analysis.png"), dpi=300, transparent=True)
    plt.show()


# %%
def plot_sentence_count(df, output_path, text_column="text", x_limit=None):
    """Generates a vertical violin plot showing the distribution of sentence count with whiskers and saves it as an image."""
    df["sentence_count"] = df[text_column].apply(lambda x: x.count(".") + x.count("!") + x.count("?"))

    plt.figure(figsize=(8, 6))
    sns.violinplot(y=df["sentence_count"], inner="box", orient="v")
    plt.title("Distribution of Sentence Count")
    plt.ylabel("Sentence Count")

    # Apply y-axis limit if specified
    if x_limit:
        plt.ylim(0, x_limit)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "sentence_count_analysis.png"), dpi=300, transparent=True)
    plt.show()


# %%
# main
if __name__ == "__main__":
    data_path = "../data/"
    output_path = "../figures/descriptive"

    # read data
    # %%
    df = pd.read_csv(f"{data_path}prepared_data.csv.gz", compression="gzip")

    # %%
    # Generate pie chart to show the source of articles
    plot_pie_chart(df, output_path, column="source")
    # %%
    # Plot top publishers
    plot_top_publishers_barplot(df, output_path, column="publisher", top_n=5)

    # %%
    # Show the average word count
    plot_word_count(df, output_path, text_column="text", x_limit=1500)
    # %%
    # show the average sentence count
    plot_sentence_count(df, output_path, text_column="text", x_limit=100)
    # %%
    # Preprocess the text data
    df["cleaned_text"] = df["translated_text"].apply(preprocess_text, additional_stopwords=["also"])

    df["topics"], topic_model = get_bert_topics(df["cleaned_text"])

    # %%
    top_n = 5  # Change this to your desired number of topics
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
