"""This script reads the newspaper data and prepare it for analysis"""

# %%
# import libraries
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from deep_translator import GoogleTranslator
from langdetect import detect
import nltk
import datetime
import matplotlib.pyplot as plt
nltk.download('punkt')



def detect_language(text):
    """
    Detect the language of the given text.

    :param text: The text to detect the language for.
    :return: The detected language code or 'error' if detection fails.
    """
    try:
        # Detect the language
        language = detect(text)
        return language
    except Exception:
        return "error"

def plot_language_distribution(df):
    """
    Plot the distribution of detected languages in a pie chart.

    :param df: DataFrame with the detected language information.
    """
    # Plot the language distribution
    language_counts = df["language"].value_counts()
    plt.figure(figsize=(8, 8))
    language_counts.plot(kind="pie", autopct="%1.1f%%", startangle=140)
    plt.title("Distribution of Detected Languages")
    plt.ylabel("")  # Hide the y-label
    plt.show()

def split_text_into_chunks(text, max_length=5000):
    """
    Split the text into chunks with a maximum length, ensuring that chunks
    are created at sentence boundaries.

    Parameters:
    - text (str): Input text.
    - max_length (int): Maximum length of each chunk.

    Returns:
    - List[str]: List of text chunks.
    """
    sentences = nltk.tokenize.sent_tokenize(text, language="italian")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding the next sentence exceeds the max_length, save the current chunk and start a new one
        if len(current_chunk) + len(sentence) + 1 > max_length:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            # Add the sentence to the current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def translate(text, source_lang=None, target_lang="en"):
    """
    Translate text to English.

    Parameters:
    - text (str): Input text.
    - source_lang (str, optional): Source language code. If None, it will be detected.
    - target_lang (str): Target language code.

    Returns:
    - str: Translated text in English or original text if translation fails.
    """
    if source_lang is None:
        source_lang = detect_language(text)
    if source_lang == "error":
        return text  # Return original text if language detection fails

    try:
        # Split text into chunks
        chunks = split_text_into_chunks(text)
        translated_chunks = [
            GoogleTranslator(source=source_lang, target=target_lang).translate(chunk) for chunk in chunks
        ]
        return " ".join(translated_chunks)
    except Exception as e:
        return text  # Return original text if translation fails


def remove_similar_texts(df, text_column="text", date_column="date", similarity_threshold=0.90):
    """
    Remove duplicate or highly similar texts based on embedding similarity and date matching.
    If date is NaN, drops duplicate texts based on the text column only.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        text_column (str): The name of the column with text data.
        date_column (str): The name of the column with date data.
        similarity_threshold (float): Threshold for cosine similarity to consider texts as duplicates.

    Returns:
        pd.DataFrame: Cleaned DataFrame with similar texts removed.
    """
    # Load the embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate embeddings for all text entries
    embeddings = model.encode(df[text_column].tolist(), convert_to_tensor=True)

    # Calculate pairwise similarity matrix
    similarity_matrix = util.cos_sim(embeddings, embeddings).numpy()

    # Set to keep track of indices to drop
    to_drop = set()

    # Loop through the similarity matrix to find duplicates
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            # Check if similarity exceeds threshold
            if similarity_matrix[i][j] >= similarity_threshold:
                # Check if the date is NaN for both entries
                if pd.isna(df[date_column][i]) and pd.isna(df[date_column][j]):
                    # Mark duplicate based on text only, keeping the first occurrence
                    if j not in to_drop:
                        to_drop.add(j)
                elif df[date_column][i] == df[date_column][j]:
                    # Compare word counts and mark the shorter text for dropping
                    if len(df[text_column][i].split()) < len(df[text_column][j].split()):
                        to_drop.add(i)
                    else:
                        to_drop.add(j)

    # Drop entries with identified indices
    df_cleaned = df.drop(to_drop).reset_index(drop=True)

    return df_cleaned


# %%
if __name__ == "__main__":
    # Set path
    path = "data/"

    # read gnews data
    gnews_file = "alluvione emilia-romagna-2023-04-01-to-2024-03-01.csv"
    gnews = pd.read_csv(f"{path}gnews/{gnews_file}")

    # %%
    # prepare gnews data
    columns = ["title", "published date", "publisher_name", "text"]
    gnews = gnews[columns]
    gnews = gnews.rename(columns={"published date": "date", "publisher_name": "publisher"})
    gnews["source"] = "gnews"
    gnews["date"] = pd.to_datetime(gnews["date"], format="%a, %d %b %Y %H:%M:%S GMT").dt.strftime("%a, %d %b %Y")

    # %%
    # read lexis-nexis articles
    ln_news = pd.read_csv(f"{path}lexis_nexis/ln_articles.csv.gz", compression="gzip")
    # prepare gnews data
    ln_news = ln_news.rename(columns={"Title": "title", "Publisher": "publisher", "Body": "text", "Load Date": "date"})
    ln_news = ln_news[["title", "date", "publisher", "text"]]
    ln_news["date"] = pd.to_datetime(ln_news["date"], format="%B %d, %Y")
    ln_news["source"] = "lexis-nexis"

    # %%
    # merge gnews and ln_news
    df = pd.concat([gnews, ln_news], ignore_index=True)

    # %%
    # drop the articles with missing text
    df = df.dropna(subset="text").reset_index(drop=True)
    # Clean the 'publisher' column
    df["publisher"] = df["publisher"].str.replace(r" \(Italy\)", "", regex=True)  # Remove ' (Italy)'
    df["publisher"] = df["publisher"].apply(lambda x: re.sub(r"[^a-zA-Z\s]", "", x))  # Remove special characters
    df["publisher"] = df["publisher"].str.lower().str.strip()  # Convert to lowercase and strip the white space

    # %%
    # translate articles into english
    # Initialize tqdm progress bar
    tqdm.pandas(desc="Translating")
    # Translate the text
    df["translated_text"] = df["text"].progress_apply(translate)

    # %%
    # drop articles with high similarity
    df_cleaned = remove_similar_texts(df, text_column="translated_text")
    df_cleaned = df_cleaned.drop_duplicates(subset=["text", "translated_text"])
    # %%
    # drop articles that are not published between 2023-04-01 and 2024-03-01
    start_date = datetime.datetime(2023, 4, 1)
    end_date = datetime.datetime(2024, 3, 1)
    df_cleaned["date"] = pd.to_datetime(df_cleaned["date"], errors="coerce")
    df_cleaned = df_cleaned[
        (df_cleaned["date"] >= start_date) & (df_cleaned["date"] <= end_date)
    ].reset_index(drop=True)
    # %%
    # Export prepared data
    df_cleaned.to_csv(f"{path}/prepared_data.csv.gz", compression="gzip", index=False)
    # %%
