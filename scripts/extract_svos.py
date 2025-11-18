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
from transformers.models.auto import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

import torch

# %%-------------------------------------------------------------------------- #
#                                    Methods                                   #
# ---------------------------------------------------------------------------- #
def clean_roles(df, preprocessor, output_path=None):
    """
    Clean ARG roles and verbs after extraction.
    Operates ONLY on available ARG columns.
    """
    # Identify role columns present
    role_cols = [c for c in ["ARG0", "B-V", "ARG1", "ARG2"] if c in df.columns]

    if not role_cols:
        print("No role columns found — skipping clean_roles()")
        return df

    # Only pass role columns to the processor, one dict per row
    roles = df[role_cols].to_dict(orient="records")

    postproc = preprocessor.process_roles(
        roles,
        max_length=100,
        progress_bar=True,
        output_path=output_path
    )

    post_df = pd.DataFrame(postproc)

    # Replace only the existing columns (keep robustness)
    for col in role_cols:
        if col in post_df.columns:
            df[col] = post_df[col]

    return df

def perform_srl(
    df,
    preprocessor,
    model_path="https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz"
):
    """
    Run SRL + (optional) SVO extraction on sentence-level DataFrame.
    Requires df to have: ["sentence", "sentence_index"].
    """

    # Safety check
    if "sentence_index" not in df.columns:
        df = df.reset_index(drop=False).rename(columns={"index": "sentence_index"})

    # -----------------------------
    # 1. RUN SRL
    # -----------------------------
    from relatio import SRL as SRLModel

    srl_model = SRLModel(
        path=model_path,
        batch_size=10,
        cuda_device=-1,
    )

    srl_res = srl_model(df["sentence"].tolist(), progress_bar=True)

    # Extract ARG roles
    srl_roles, srl_idx = extract_roles(
        srl_res,
        used_roles=["ARG0", "B-V", "B-ARGM-NEG", "B-ARGM-MOD", "ARG1", "ARG2"],
        only_triplets=False,
        progress_bar=True,
    )

    srl_df = pd.DataFrame(srl_roles)
    srl_df["sentence_index"] = srl_idx

    # Merge SRL roles
    df = df.merge(srl_df, on="sentence_index", how="left")

    # -----------------------------
    # 2. RUN SVO extraction
    # -----------------------------
    svo_idx, svo_roles = preprocessor.extract_svos(
        df["sentence"].tolist(),
        expand_nouns=True,
        only_triplets=True,
        progress_bar=True
    )

    svo_df = pd.DataFrame(svo_roles)
    svo_df["sentence_index"] = svo_idx

    # Merge SVO roles (with suffix "_svo")
    df = df.merge(svo_df, on="sentence_index", how="left", suffixes=("", "_svo"))

    return df

def set_preprocessor(spacy_model="en_core_web_sm", add_stop_words=None, n_process=-1):
    """Initialize a relatio Preprocessor with custom stopwords."""

    stop_words = set(stopwords.words("english"))

    if add_stop_words:
        stop_words.update(add_stop_words)

    p = Preprocessor(
        spacy_model=spacy_model,
        remove_punctuation=True,
        remove_digits=True,
        lowercase=True,
        lemmatize=True,
        remove_chars=[
            '"', "-", "^", ".", "?", "!", ";", "(", ")", ",", ":", "'", "+", "&",
            "|", "/", "{", "}", "~", "_", "`", "[", "]", ">", "<", "=", "*", "%",
            "$", "@", "#", "’", "\n",
        ],
        stop_words=stop_words,
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

def load_local_roberta():
    model_path = ROOT / "models/roberta-sentiment/" 

    # auto-detects vocab.json + merges.txt (BPE tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    return tokenizer, model

def compute_sentiment(df_sentences, tokenizer, model):
    """
    Compute sentiment scores for unique sentences using a local transformer model.
    Expects df_sentences to have columns: ["sentence_global_id", "sentence"].
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # RoBERTa typically has max_position_embeddings = 514
    max_pos = getattr(model.config, "max_position_embeddings", 512)

    # For RoBERTa: we keep a small safety margin for special tokens
    # (CLS + SEP) → so use max_length = max_pos - 2
    if getattr(model.config, "model_type", "") == "roberta":
        max_length = max_pos - 2
    else:
        max_length = max_pos

    # 1) unique sentences only
    unique = df_sentences[["sentence_global_id", "sentence"]].drop_duplicates()

    labels = []
    scores = []

    sentences = unique["sentence"].tolist()
    batch_size = 32

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]

        # Explicit max_length avoids the 514/515 issue
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)

        probs = out.logits.softmax(dim=1)

        # Handle both 2-label (neg/pos) and 3-label (neg/neu/pos) models
        num_labels = probs.shape[1]

        batch_labels = probs.argmax(dim=1).cpu().numpy()
        if num_labels == 3:
            # score = P(pos) - P(neg)
            batch_scores = (probs[:, 2] - probs[:, 0]).cpu().numpy()
        elif num_labels == 2:
            # binary: [neg, pos]
            batch_scores = (probs[:, 1] - probs[:, 0]).cpu().numpy()
        else:
            # Fallback: use argmax probability as score
            batch_scores = probs.max(dim=1).values.cpu().numpy()

        labels.extend(batch_labels)
        scores.extend(batch_scores)

    unique["sentiment_label"] = labels
    unique["sentiment_score"] = scores

    return unique






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

    # %% clean the roles - preprocess text
    sentences = clean_roles(sentences, p)

    #%% drop non triplets
    sentences = sentences.dropna(subset=['ARG0', 'B-V', 'ARG1'])

    #%% generate SVO index
    sentences.loc[:, 'svo_id'] = np.arange(len(sentences))

    #%%  merge df with sentences
    df.rename(columns={'article_index':'doc_id'}, inplace=True)
    df.drop(columns=['id', 'translated_text'], inplace=True)
    sentences = pd.merge(left=sentences, right=df, on='doc_id')

    #%% TODO Perform sentiment analysis
    
    # Load local model
    tokenizer, model = load_local_roberta()

    # Compute sentiment on unique sentences
    sentiment_df = compute_sentiment(sentences, tokenizer, model)

    sentences = sentences.merge(sentiment_df, on="sentence_global_id", how="left")


    # %% export svos
    sentences.to_csv(
        f"{file_path}/news_corpus_svo.csv.gz", compression="gzip", index=False
    )

# %%
