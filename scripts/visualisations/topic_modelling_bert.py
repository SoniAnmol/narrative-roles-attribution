""" This script contains code to perform topic modelling using BERT algorithm."""

# ---------------------------------------------------------------------------- #
# import libraries
from topic_modelling_lda import preprocess_text
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------- #
# define methods
def merge_similar_topics(docs, topic_model):
    # Merge similar topics using reduce_topics

    topic_model = topic_model.reduce_topics(docs=docs, nr_topics=20)

    # Get the reduced topics and probabilities

    merged_topics = topic_model.topics_

    reduced_probabilities = topic_model.probabilities_
    topic_info = topic_model.get_topic_info()
    return merged_topics, topic_info


def get_bert_topics(text):
    if type(text) is not list:
        text = text.to_list()
    # Initialize the sentence transformer model
    sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # Extract embeddings
    embeddings = sentence_model.encode(text, show_progress_bar=True)

    # Initialize BERTopic model
    topic_model = BERTopic()

    # Fit the model on the data
    topics, probabilities = topic_model.fit_transform(text, embeddings)

    # Get the topic descriptions
    topic_info = topic_model.get_topic_info()
    print(f"Number of topics: {len(topic_info)}")
    if len(topic_info) >= 10:
        print("Too many topics. Reducing topics by merging similar topics")
        topics, topic_info = merge_similar_topics(text, topic_model=topic_model)

    fig = topic_model.visualize_documents(text, embeddings=embeddings)
    fig.write_html("topics.html")

    # Visualize the topic distribution with horizontal bar plots
    plot_topic_distribution(topic_info)

    return (
        topics,
        topic_model,
    )


def plot_topic_distribution(topic_info):
    # Visualize the topic distribution with horizontal bar plots
    plt.figure(figsize=(10, 6))
    sns.barplot(y=topic_info.Name, x=topic_info.Count, palette="viridis")
    plt.title("Distribution of Topics")
    plt.xlabel("Number of Documents")
    plt.ylabel("Topic")
    plt.show()


# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    # read file
    df = pd.read_csv("../data/alluvione emilia-romagna-2023-04-01-to-2024-03-01-translated.csv")

    # Ensure the date column is in datetime format
    df["date"] = pd.to_datetime(df["published date"])
    # Sort the dataframe by date
    df = df.sort_values(by="date")
    # Preprocess the text data
    df["cleaned_text"] = df["translated_text"].apply(preprocess_text, additional_stopwords=["also"])

    df["topics"], topic_model = get_bert_topics(df["cleaned_text"])
