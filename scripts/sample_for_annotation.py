""""This script prepares the training data for annotation"""

# %%
import pandas as pd
import numpy as np
import json
from pathlib import Path
from extract_svos import apply_final_mapping
# %% methods

def sample_top_sentence_indices(
    df, arg0_col="ARG0", arg1_col="ARG1", sentence_index_col="sentence_index", top_n=10, actor_categories=None
):
    # Initialize an empty DataFrame to store the sampled rows
    sampled_df = pd.DataFrame()

    # Keep track of sentence_index values already added
    sampled_sentence_indices = set()

    # Get unique values of ARG0 and ARG1 (assuming they are the same)
    if actor_categories:
        unique_values = actor_categories
    else:
        unique_values = df[arg0_col].unique()

    for value in unique_values:
        # Filter rows where ARG0 or ARG1 equals the current unique value
        filtered_rows = df[(df[arg0_col] == value) | (df[arg1_col] == value)]

        # Exclude rows with already sampled sentence_index
        filtered_rows = filtered_rows[~filtered_rows[sentence_index_col].isin(sampled_sentence_indices)]

        # Select top 15 unique sentence_index rows
        sampled_rows = filtered_rows.drop_duplicates(subset=sentence_index_col).head(top_n)

        # Add sampled sentence_index values to the set
        sampled_sentence_indices.update(sampled_rows[sentence_index_col])

        # Append sampled rows to the result DataFrame
        sampled_df = pd.concat([sampled_df, sampled_rows])

    return sampled_df

def get_actor_count(df, actor=None, col1='ARG0', col2='ARG1'):
    if actor is None:
        actors = pd.read_csv('../data/actor_directory/actor_directory.csv')
        # Get the list of unique actors
        actor = list(actors.actor.unique())
        actor.remove(np.nan)

    # Get the distribution of actors in the text corpus
    entities = pd.concat([df[col1], df[col2]])
    entities = entities[entities.isin(actor)]
    actor_count = entities.value_counts()
    return actor_count

def build_svo_key(df):
    """
    Build a duplicate-safe unique key for each SVO.
    """
    return (
        df["sentence"].astype(str) + "|" +
        df["ARG0_raw"].astype(str) + "|" +
        df["B-V"].astype(str) + "|" +
        df["ARG1_raw"].astype(str)
    )

def compute_sampling_needs(actor_count_sample, actor_count_corpus, min_target=50):
    needs = {}
    for actor, _ in actor_count_corpus.items():
        current = actor_count_sample.get(actor, 0)
        target = min_target
        need = max(target - current, 0)
        if need > 0:
            needs[actor] = need
    return needs

# def stratified_sample_next_batch(df, actors_df, sampled_df, needs_dict):
#     sampled_keys = set(build_svo_key(sampled_df))

#     # Build mapping: actor_name → category_label
#     mapping = dict(zip(actors_df.actor, actors_df.category))

#     df = df.copy()
#     df["actor_category"] = (
#         df["ARG0"].map(mapping).fillna(df["ARG1"].map(mapping))
#     )

#     df.rename(columns={'sentences':'sentence'}, inplace=True)

#     next_batch = []

#     for actor_cat, n_needed in needs_dict.items():

#         subset = df[df["actor_category"] == actor_cat].copy()
#         print(actor_cat, "→ subset before removing sampled:", len(subset))

#         if len(subset) == 0:
#             continue

#         subset["svo_key"] = build_svo_key(subset)

#         subset = subset[~subset["svo_key"].isin(sampled_keys)]
#         print(actor_cat, "→ subset after removing sampled:", len(subset))

#         if len(subset) == 0:
#             continue

#         subset = subset.drop_duplicates(subset=["svo_key"])

#         take_n = min(n_needed, len(subset))
#         next_batch.append(subset.sample(take_n, random_state=42))

#     if not next_batch:
#         return pd.DataFrame()

#     return pd.concat(next_batch, ignore_index=True)

def stratified_sample_next_batch(df, actors_df, sampled_df, needs_dict, final_mapping):
    df = df.copy()

    # Normalize for matching
    df["ARG0_norm"] = df["ARG0"].astype(str).str.strip().str.lower()
    df["ARG1_norm"] = df["ARG1"].astype(str).str.strip().str.lower()

    # Apply final mapping to get canonical label (actor OR category OR cluster label)
    df["arg0_final"] = df["ARG0_norm"].map(final_mapping)
    df["arg1_final"] = df["ARG1_norm"].map(final_mapping)

    # This is your "actor_category"
    df["actor_category"] = df["arg0_final"].fillna(df["arg1_final"])

    # Build list of sampled SVO keys
    sampled_keys = set(build_svo_key(sampled_df))
    df["svo_key"] = build_svo_key(df)

    df.rename(columns={'sentences': 'sentence'}, inplace=True)

    next_batch = []

    for actor_cat, n_needed in needs_dict.items():
        subset = df[df["actor_category"] == actor_cat].copy()
        print(actor_cat, "→ subset before removing sampled:", len(subset))

        if len(subset) == 0:
            continue

        subset = subset[~subset["svo_key"].isin(sampled_keys)]
        print(actor_cat, "→ subset after removing sampled:", len(subset))

        if len(subset) == 0:
            continue

        subset = subset.drop_duplicates(subset=["svo_key"])

        take_n = min(n_needed, len(subset))
        next_batch.append(subset.sample(take_n, random_state=42))

    if not next_batch:
        return pd.DataFrame()

    return pd.concat(next_batch, ignore_index=True)


# %% main
if __name__ == "__main__":
    # * Read files
    ROOT = Path(__file__).resolve().parent.parent 
    df = pd.read_csv("../data/news_corpus/news_corpus_svo.csv.gz")
    actors = pd.read_csv('../data/actor_directory/actor_directory.csv')
    sampled_data = pd.read_excel(f"{ROOT}/data/training_data/annotated_training_data.xlsx")
    with open(f"{ROOT}/data/actor_directory/actor_mapping.json", "r", encoding="utf-8") as f:
        final_mapping = json.load(f)
    
    sampled_data = apply_final_mapping(df=sampled_data, final_mapping=final_mapping)

    entity_dict = dict(zip(actors['entity'], actors['category']))
    actors = actors[actors.keep == 0].copy()
    actor_dict = dict(zip(actors['category'], actors['actor']))

    for col in ['ARG0', 'ARG1']:
        # preserve raw values 

        sampled_data.loc[:, col] = sampled_data[col].replace(entity_dict)
        sampled_data.loc[:, col] = sampled_data[col].replace(actor_dict)


    #%% Get the distribution of actors in the text corpus
    actor_count_corpus = get_actor_count(df, col1='ARG0', col2 = 'ARG1')
    actor_count_sample = get_actor_count(sampled_data, col1='ARG0', col2 = 'ARG1')
    
    # sampled actor vs actors in corpus
    actor_count = pd.merge(actor_count_sample, actor_count_corpus, left_index=True, right_index=True, how='outer', suffixes=('_sample', '_corpus'))
    actor_count['percentage_sampled'] = actor_count['count_sample']/actor_count['count_corpus']
    print(actor_count)

    #%% compute sampling needs
    needs_dict = compute_sampling_needs(actor_count_sample, actor_count_corpus, min_target=100)
    print("Additional samples needed:", needs_dict)

    # clean corpus so actor categories are consistent
    actors = actors[actors.keep == 0]
    actor_list = list(actors.actor.unique())

    # Filter corpus for relevant SVOs
    df_corpus = df[
    df["ARG0"].isin(actor_list) | df["ARG1"].isin(actor_list)].copy()

    #%%
    df_corpus.rename(columns={'sentences':'sentence'}, inplace=True)
    next_batch = stratified_sample_next_batch(
        df=df_corpus,
        actors_df=actors,
        sampled_df=sampled_data,
        needs_dict=needs_dict,
        final_mapping=final_mapping )

    # %% Filter actors of interest
    undersampled_actors = [
    'environment',
    'essential goods and infrastructure',
    'municipality',
    'business',
    'national government',
    'people',
    'political actors',
    'agriculture']

    num_samples = 30
    additional_samples = []

    # Normalize sentence text for robust duplicate detection
    def normalize_sentence(s):
        if isinstance(s, str):
            return s.strip().lower()
        return ""

    # Build set of already annotated sentences
    used_sentences = set(sampled_data["sentence"].apply(normalize_sentence))

    for actor in undersampled_actors:
        # Mask for this actor
        mask = (df_corpus["ARG0"] == actor) | (df_corpus["ARG1"] == actor)

        # Filter sentences for this actor
        candidate_pool = df_corpus[mask].copy()

        # Normalize
        candidate_pool["norm_sentence"] = candidate_pool["sentence"].apply(normalize_sentence)

        # Drop already used sentences
        candidate_pool = candidate_pool[~candidate_pool["norm_sentence"].isin(used_sentences)]

        # Choose up to num_samples
        additional_sample = candidate_pool.head(num_samples)

        additional_samples.append(additional_sample)

# Combine and remove duplicates based on normalized sentence text
    next_batch = (
        pd.concat(additional_samples)
        .assign(norm_sentence=lambda df: df["sentence"].apply(normalize_sentence))
        .drop_duplicates(subset=["norm_sentence"])
        .drop(columns=["norm_sentence"])
    )

    #%%
    next_batch["narratives"] = (
    next_batch["ARG0"] + " " +
    next_batch["B-V"] + " " +
    next_batch["ARG1"])

    next_batch['svo_type'] = 'triplet'

    cols = ['doc_id', "sentence", "title", "date", "publisher", "text",
    "doc", "source", "sentence_id", "ARG0", "ARG1", "sentiment_score",
    "ARG0_raw",	"B-V", "ARG1_raw", "narratives", "svo_type"]

    next_batch = next_batch.loc[:, cols].copy()
    # %%
    # * Export sample dataset for annotation
    next_batch.to_excel("../data/training_data/sampled_data_for_annotation.xlsx", index=False)

# %%
