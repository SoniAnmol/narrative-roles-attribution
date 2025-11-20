# Extract SVOs
"""This script splits the 'cleaned' text data into sentences and for each sentence, extract the Subject-Verb-Object structures"""

#  %%------------------------------------------------------------------------- #
#                                  Import libraries                            #
# ---------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from nltk.corpus import stopwords
from relatio import Preprocessor
from relatio import extract_roles
import datetime
import spacy
import torch
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import hdbscan
from transformers.models.auto import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)



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

def cluster_entities_semantically(
    sentences_df, 
    arg_cols=["ARG0", "ARG1"], 
    k=None,
    k_min=10, 
    k_max=80,
    step=5,
    model_name="all-MiniLM-L6-v2"
):
    """
    Automatically selects optimal number of clusters (k) using silhouette score  
    unless k is manually supplied.

    Returns:
        - entity_cluster_map (entity → cluster_id)
        - cluster_label_map (cluster_id → representative label)
    """

    # -------------------------------------------------------
    # 1. Extract unique canonical entities
    # -------------------------------------------------------
    entities = (
        sentences_df[arg_cols]
        .melt(value_name="entity")["entity"]
        .dropna()
        .astype(str)
    )

    clean_entities = sorted(list(set([e.strip() for e in entities if e.strip()])))

    if len(clean_entities) == 0:
        print("No valid entities found for clustering.")
        return {}, {}

    print(f"Embedding {len(clean_entities)} entities...")

    # -------------------------------------------------------
    # 2. Compute embeddings
    # -------------------------------------------------------
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(clean_entities, convert_to_numpy=True, show_progress_bar=True)

    # -------------------------------------------------------
    # 3. Automatically select k if not provided
    # -------------------------------------------------------
    if k is None:
        print(f"\nSelecting optimal number of clusters between {k_min} and {k_max}...")

        best_k = None
        best_score = -1

        for curr_k in range(k_min, min(k_max, len(clean_entities)) + 1, step):
            kmeans = KMeans(n_clusters=curr_k, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(embeddings)

            # Silhouette requires at least 2 clusters
            if len(set(labels)) < 2:
                continue

            score = silhouette_score(embeddings, labels)

            print(f"k={curr_k}: silhouette={score:.4f}")

            if score > best_score:
                best_k = curr_k
                best_score = score

        k = best_k or k_min  # fallback

        print(f"\nOptimal k selected = {k}  (silhouette={best_score:.3f})")

    else:
        print(f"Using user-defined k = {k}")

    # -------------------------------------------------------
    # 4. Final clustering with optimal k
    # -------------------------------------------------------
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    cluster_ids = kmeans.fit_predict(embeddings)

    entity_cluster_map = dict(zip(clean_entities, cluster_ids))

    # -------------------------------------------------------
    # 5. Compute entity frequencies in corpus
    # -------------------------------------------------------
    freq_df = (
        sentences_df[arg_cols]
        .melt(value_name="entity")
        .dropna()
        .astype(str)
        .groupby("entity")
        .size()
        .reset_index(name="freq")
    )

    freq_df["cluster_id"] = freq_df["entity"].map(entity_cluster_map)

    # -------------------------------------------------------
    # 6. Cluster representative = most frequent term
    # -------------------------------------------------------
    cluster_label_map = (
        freq_df.sort_values(["cluster_id", "freq"], ascending=[True, False])
        .groupby("cluster_id")
        .first()["entity"]
        .to_dict()
    )

    # print("Clustering + labeling completed.")
    return entity_cluster_map, cluster_label_map

def export_clusters_to_excel(
    df,
    cluster_map,
    entity_cols=["ARG0", "ARG1"],   # <— now supports BOTH
    output_path="clusters.xlsx"
):
    """
    Export clusters to Excel, combining entity frequencies from multiple columns
    (e.g., ARG0 + ARG1). 
    
    Output:
        - One column per cluster
        - Entities inside each cluster sorted by frequency (ARG0 + ARG1)
        - Format: 'entity (count)'
    """

    # -------------------------------------------------
    # 1. Combine ARG0 + ARG1 into long format
    # -------------------------------------------------
    long_df = pd.concat(
        [df[col].dropna().astype(str).to_frame("entity") for col in entity_cols],
        axis=0,
        ignore_index=True
    )

    # Normalize entity names (optional but helps matching)
    long_df["entity_clean"] = long_df["entity"].str.lower().str.strip()

    # -------------------------------------------------
    # 2. Map clusters
    # -------------------------------------------------
    long_df["cluster_id"] = long_df["entity_clean"].map(cluster_map)

    # Remove missing clusters
    long_df = long_df.dropna(subset=["cluster_id"])

    # -------------------------------------------------
    # 3. Count frequencies
    # -------------------------------------------------
    freq = (
        long_df.groupby(["cluster_id", "entity_clean"])
               .size()
               .reset_index(name="count")
    )

    # -------------------------------------------------
    # 4. Build cluster → entity list column dict
    # -------------------------------------------------
    cluster_columns = {}

    for cid in sorted(freq["cluster_id"].unique()):
        sub = freq[freq["cluster_id"] == cid]

        # Sort by descending frequency
        sub = sub.sort_values("count", ascending=False)

        formatted = [
            f"{row['entity_clean']} ({row['count']})"
            for _, row in sub.iterrows()
        ]

        cluster_columns[f"cluster_{int(cid)}"] = formatted

    # -------------------------------------------------
    # 5. Turn into a DataFrame with uneven column lengths
    # -------------------------------------------------
    cluster_df = pd.DataFrame(dict([
        (col, pd.Series(vals)) for col, vals in cluster_columns.items()
    ])).fillna("")

    # -------------------------------------------------
    # 6. Save to Excel
    # -------------------------------------------------
    cluster_df.to_excel(output_path, index=False)
    print(f"Cluster export created at:\n{output_path}")

    return cluster_df


# %%-------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent 
    # * Read data
    file_path = ROOT / "data/news_corpus/"
    file_name = "/news_corpus.csv.gz"
    output_path = ROOT / "figures/descriptive/"
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

    #%% Perform sentiment analysis
    # Load local model
    tokenizer, model = load_local_roberta()
    # Compute sentiment on unique sentences
    sentiment_df = compute_sentiment(sentences, tokenizer, model)
    sentences = sentences.merge(sentiment_df, on="sentence_global_id", how="left")


    # %% export svos
    # sentences.to_csv(
    #     f"{file_path}/news_corpus_svo.csv.gz", compression="gzip", index=False
    # )

    # %% Read sentences_svo
    # sentences = pd.read_csv(f"{file_path}/news_corpus_svo.csv.gz", compression="gzip")

    # %% Drop too long and too short articles
     sentences = drop_long_documents(sentences, sentence_col="sentence_x", threshold=1000)
    sentences = drop_short_documents(sentences,min_words=100)
    # %%  Plot the top publisher sources 
    plot_articles_by_category_and_source(sentences, file_name=f"{output_path}/articles_by_category_and_source.png")
    # Generate a table of top publishers
    plot_top_publishers_barplot(sentences, output_path, top_n=10)

    #%% Show the average word count
    print(sentences.groupby('source')['doc_id'].nunique())
    print(f"Total Articles = {sentences.doc_id.nunique()}")
    print(f"Total senteces in the corpus =  {sentences.sentence_global_id.nunique()}")
    print(f"Toal SVOs = {sentences.svo_id.nunique()}")
    plot_word_count(sentences, output_path, x_limit=1000)
    plot_sentence_count(sentences, output_path, x_limit=80)
    plot_words_per_sentence(sentences, output_path=output_path)
    
    # %% drop redundant column
    sentences.drop(columns=['sentence_y'], inplace=True)
    sentences.rename(columns={'sentence_x':'sentences'}, inplace=True)

    # %% TODO improve the actor clustering

    # %%  Read actor directory
    actor_directory_path = ROOT / "data/actor_directory"
    actor_directory = pd.read_csv(f"{actor_directory_path}/actor_directory.csv")
    
    entitiy_dict = dict(zip(actor_directory['entity'], actor_directory['category']))
    actor_directory = actor_directory[actor_directory.keep == 0].copy()
    actor_dict = dict(zip(actor_directory['category'], actor_directory['actor']))

    # %% Apply actor and entity directory
    for col in ['ARG0', 'ARG1']:
        sentences.loc[:, col] = sentences[col].replace(entitiy_dict)
        sentences.loc[:, col] = sentences[col].replace(actor_dict)


    # %% semantic clustering
    all_entities = pd.concat([
        sentences["ARG0"],
        sentences["ARG1"]
    ]).dropna().unique()

    # embed and cluster once
    entity_cluster_map, cluster_label_map  = cluster_entities_semantically(sentences, arg_cols=["ARG0", "ARG1"], k=30)

    cluster_df = export_clusters_to_excel(
    df=sentences,
    cluster_map=entity_cluster_map,
    output_path= ROOT / "data/actor_directory/actor_clusters.xlsx")


    #%% map both columns
    sentences["ARG0"] = sentences["ARG0"].map(entity_cluster_map)
    sentences["ARG1"] = sentences["ARG1"].map(entity_cluster_map)



    #%%  TODO Map ARG0 & ARG1 with actor_directory
    # for col in ['ARG0', 'ARG1']:
    #     df_cleaned.loc[:, col] = df_cleaned[col].replace(actor_dict)
    
