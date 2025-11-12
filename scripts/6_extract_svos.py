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
    df,
    preprocessor,
    spacy_model="en_core_web_sm",
):
    """Splits the text data into multiple sentences
    Args:
    df: A pandas data frame with columns 'doc' containing text data
    Returns:
    df: DataFrame with processed sentences with 'id' representing doc id and sentence_id
    """
    # check for 'id' in df columns; if not create a new column called 'id'
    if "id" not in df.columns:
        # create a unique id for all the articles
        df["id"] = np.arange(len(df))

    # split into sentences
    df_sentences = preprocessor.split_into_sentences(df, output_path=None, progress_bar=True)
    # combine df of sentences with original df
    df = pd.merge(df_sentences, df, on="id")  # type: ignore
    # create an index for mapping sentences
    df["sentence_index"] = np.arange(len(df))
    return df


# %%-------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":

    # * Read data
    file_path = "data/prepared_data.csv.gz"
    df = pd.read_csv(file_path)

    # %%
    # * Split text data into sentences
    p = set_preprocessor()
    df["doc"] = df["translated_text"]
    # %%
    df = split_sentences(df, p)

    # %%
    # * Semantic Role Labelling
    df = perform_srl(df, p)

    # %%
    # * clean the roles - preprocess text
    df = clean_roles(df, p)

    # %%
    # * export svos
    df.to_csv(
        f"output/svo_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv.gz", compression="gzip", index=False
    )
