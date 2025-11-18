# Extract SVOs
"""This script splits the 'cleaned' text data into sentences and for each sentence, extract the Subject-Verb-Object structures"""

#  %%------------------------------------------------------------------------- #
#                                  Import libraries                            #
# ---------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from relatio import Preprocessor
from relatio import extract_roles
import datetime
import spacy
from pathlib import Path


# %%-------------------------------------------------------------------------- #
#                                    Methods                                   #
# ---------------------------------------------------------------------------- #
def clean_roles(df, preprocessor, output_path=None):
    """Clean the roles and verbs"""
    roles = df.apply(lambda row: {col: row[col] for col in df.columns if pd.notna(row[col])}, axis=1).tolist()
    postproc_roles = preprocessor.process_roles(roles, max_length=100, progress_bar=True, output_path=output_path)
    df.loc[:, ["ARG0", "B-V", "ARG1"]] = pd.DataFrame(postproc_roles)[["ARG0", "B-V", "ARG1"]]
    return df

def perform_srl(
    df, preprocessor, model_path="https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz"
):
    """
    Performs semantic role labelling on the sentences
    """
    # create an instance of SRL class
    from relatio import SRL

    SRL = SRL(
        path=model_path,
        batch_size=10,
        cuda_device=-1,
    )
    # Perform SRL
    srl_res = SRL(df["sentence"], progress_bar=True)

    # extract roles from srl results
    roles, sentence_index = extract_roles(
        srl_res,
        used_roles=["ARG0", "B-V", "B-ARGM-NEG", "B-ARGM-MOD", "ARG1", "ARG2"],
        # used_roles=["ARG0", "B-V", "B-ARGM-NEG", "ARG1", "ARG2"],
        only_triplets=True,
        progress_bar=True,
    )
    # Extract SVO structures form SRL results
    sentence_index, roles = preprocessor.extract_svos(
        df["sentence"].to_list(), expand_nouns=True, only_triplets=True, progress_bar=True
    )

    # update svos in df
    roles_df = pd.DataFrame(roles)
    roles_df["sentence_index"] = sentence_index
    df = pd.merge(df, roles_df, on="sentence_index")

    return df

def set_preprocessor(spacy_model="en_core_web_sm", add_stop_words=None, n_process=-1):
    """ "Sets the preprocessor for cleaning the text
    Args:
    spacy_model: Spacy model to be used for cleaning the text
    add_stop_words: list of additional stop words
    Returns:
    p: an instance of preprocessor class
    """
    #  Define stop-words to remove from sentences
    stop_words = set(stopwords.words("english"))

    if add_stop_words is not None:
        stop_words.update(add_stop_words)

    # initialise preprocessor to clean the sentences
    p = Preprocessor(
        spacy_model=spacy_model,
        remove_punctuation=True,
        remove_digits=True,
        lowercase=True,
        lemmatize=True,
        remove_chars=[
            '"',
            "-",
            "^",
            ".",
            "?",
            "!",
            ";",
            "(",
            ")",
            ",",
            ":",
            "'",
            "+",
            "&",
            "|",
            "/",
            "{",
            "}",
            "~",
            "_",
            "`",
            "[",
            "]",
            ">",
            "<",
            "=",
            "*",
            "%",
            "$",
            "@",
            "#",
            "’",
            "\n",
        ],
        stop_words=stop_words,  # type: ignore
        n_process=n_process,
    )

    return p

def split_sentences(
    df: pd.DataFrame,
    text_col: str = "doc",
    id_col: str = "id",
    spacy_model: str = "en_core_web_sm"
) -> pd.DataFrame:
    """
    Split long documents into individual sentences using spaCy's sentence segmenter.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing one document per row.
    text_col : str
        Column name containing the full raw text.
    id_col : str
        Column name to treat as document ID. If missing, IDs are created.
    spacy_model : str
        spaCy model to use for sentence segmentation.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with:
        - doc_id (original document ID)
        - sentence_id (index of sentence within the doc)
        - sentence (the extracted sentence)
        - sentence_global_id (unique index over all sentences)
    """

    # Ensure ID column exists
    if id_col not in df.columns:
        df[id_col] = np.arange(len(df))

    nlp = spacy.load(spacy_model, disable=["ner", "tagger"])
    nlp.add_pipe("sentencizer")  # ensures reliable sentence splitting

    all_sentences = []
    for doc_id, text in zip(df[id_col], df[text_col]):
        if not isinstance(text, str):
            continue
        doc = nlp(text)

        for i, sent in enumerate(doc.sents):
            clean = sent.text.strip()
            if clean:  # skip empty lines
                all_sentences.append({
                    "doc_id": doc_id,
                    "sentence_id": i,
                    "sentence": clean
                })

    # Build final DataFrame
    out = pd.DataFrame(all_sentences)
    out["sentence_global_id"] = np.arange(len(out))

    return out


# %%-------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent 
    # * Read data
    file_path = ROOT / "data/news_corpus/"
    file_name = "/news_corpus.csv.gz"
    df = pd.read_csv(f'{file_path}{file_name}')

    # %%
    # * Split text data into sentences
    p = set_preprocessor()
    df["doc"] = df["translated_text"]
    # %%
    sentences = split_sentences(df, text_col="doc")

    # %%
    # * Semantic Role Labelling
    sentences = perform_srl(sentences, p)

    # %% TODO fix and then run
    # * clean the roles - preprocess text
    df = clean_roles(df, p)

    #%% TODO merge df with senteces
    

    # %%
    # * export svos
    df.to_csv(
        f"{file_path}/news_corpus_svo.csv.gz", compression="gzip", index=False
    )
