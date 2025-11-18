"""
This script contains methods for identifying named entities in the text data.

Following are the nomenclature for the named entities.
GPE : Geo Political Entity
DATE
ORG : Organization
TIME
LOC : Location
CARDINAL
ORDINAL
PERSON
FAC : Facility
NORP : Nationalities or Religious or Political Groups
PERCENT
MONEY
LAW
LANGUAGE
"""

# ----------------------------- Import libraries ----------------------------- #
import pandas as pd
from tqdm import tqdm
import spacy
from collections import defaultdict, Counter
from translate_articles import detect_language
import re
from nltk.corpus import stopwords


# ------------------------------ Define methods ------------------------------ #
def clean_text(text):
    # List of words to remove (e.g., the, of, etc.)
    words_to_remove = set(stopwords.words("english"))

    words_to_remove.update(["region", "via"])

    # Convert the text to lowercase
    text = text.lower()

    # Remove specific words like 'the', 'of', etc.
    text = " ".join([word for word in text.split() if word not in words_to_remove])

    # Remove special characters and replace them with a space
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# def clean_text(text):
#     # Convert to lowercase
#     text = text.lower()
#     # Remove special characters
#     text = re.sub(r"[^a-z0-9\s]", " ", text)
#     return text


def count_named_entities(ents):
    """
    Count the unique named entities in each label category using the spaCy library.

    Parameters:
        ents (list): A list of dictionaries containing named entities and their labels.
                     Example: [{'Mr. Beast': 'PER', 'YouTube': 'ORG', 'USA': 'LOC'}]

    Returns:
        dict: A dictionary where keys are entity labels and values are dictionaries of
              entity counts for each label.
    """
    # Initialize a defaultdict to store the counts
    label_counts = defaultdict(int)
    entity_counts = defaultdict(Counter)

    # Iterate over each dictionary in the 'entities' column
    for entity_dict in tqdm(ents, total=len(ents), desc="Counting named entities"):
        for entity, label in entity_dict.items():
            label_counts[label] += 1
            entity_counts[label][entity] += 1

    label_counts = dict(label_counts)
    entity_counts = {label: dict(counter) for label, counter in entity_counts.items()}

    return entity_counts


def create_entities_table(entity_dict):
    """
    Create a DataFrame from the entity dictionary containing named entities and their counts.

    Parameters:
        entity_dict (dict): A dictionary with entity labels as keys and dictionaries of
                            entity counts as values.

    Returns:
        DataFrame: A pandas DataFrame containing entities, their counts, and categories.
    """
    # Count named entities
    named_entities_count = []
    for key in entity_dict.keys():
        df = (
            pd.DataFrame(list(entity_dict[key].items()), columns=["entity", "count"])
            .sort_values(by="count", ascending=False)
            .reset_index(drop=True)
        )
        df.loc[:, "category"] = key
        named_entities_count.append(df)

    named_entities_count = pd.concat(named_entities_count, axis=0)
    return named_entities_count


def extract_ents(doc):
    """
    Extract named entities from a spaCy document.

    Parameters:
        doc (spacy.tokens.doc.Doc): A spaCy document object.

    Returns:
        dict: A dictionary with entity text as keys and entity labels as values.
    """
    ents_dict = {}
    for ents in doc.ents:
        ents_dict[ents.text] = ents.label_
    return ents_dict


def perform_ner(df, text_column, nlp_model="en_core_web_trf"):
    """
    Perform Named Entity Recognition (NER) on a DataFrame and return a table of named entities with their counts.

    Parameters:
        df (DataFrame): A pandas DataFrame containing the text data.
        text_column (str): The name of the column in the DataFrame containing text to be processed.
        nlp_model (str): The name of the spaCy language model to be used. Default is "en_core_web_trf".

    Returns:
        DataFrame: A pandas DataFrame containing named entities, their counts, and categories,
                   filtered to include only entities appearing more than 10 times.
    """
    # Load language model
    nlp = spacy.load(nlp_model)

    # Pass text to the language model
    tqdm.pandas(desc="Tokenizing")
    df.loc[:, "doc"] = df[text_column].progress_apply(nlp)

    # Extract named entities
    tqdm.pandas(desc="Looking for named entities")
    df.loc[:, "ents"] = df.doc.progress_apply(extract_ents)

    named_entities = count_named_entities(df.ents)

    named_entities_df = create_entities_table(named_entities)

    # Clean the entity name
    named_entities_df["entity"] = named_entities_df["entity"].apply(clean_text)

    # remove null values
    named_entities_df = named_entities_df[named_entities_df["entity"] != ""]

    named_entities_df = named_entities_df.groupby("entity", as_index=False).agg(
        {
            "count": "sum",  # Sum the 'count' column
            "category": "first",  # Replace with relevant columns; use 'first' or another method as needed
        }
    )

    return named_entities_df


# ----------------------------------- main ----------------------------------- #
# %%
if __name__ == "__main__":
    # * Read data
    # file_path = "data/videos_on_alluvione emilia-romagna_translated_cleaned.csv"
    file_path = "data/alluvione_emilia-romagna_2023-04-01-to-2024-03-01_summarized_translated_cleaned.csv"
    # %%
    df = pd.read_csv(file_path)
    # %%
    # * Filter the text which is not translated
    df["translated_lang"] = df["translated_text"].apply(detect_language)
    df = df[df.translated_lang == "en"].reset_index(drop=True)
    # %%
    # * Perform NER
    df_ner = perform_ner(df=df, text_column="translated_text")
    # %%
    # * Store the named entities table
    df_ner.to_excel("output/named_entities.xlsx", index=False)
