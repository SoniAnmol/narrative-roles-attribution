"""This script evaluates the fit of labelled entities in forming the narratives and generates new clusters for the unclustered entities"""

# %% import libraries
import pandas as pd
from relatio.narrative_models import NarrativeModel
from relatio.utils import prettify
import datetime
import numpy as np


# %% define methods
def clean_output(narratives):
    # Drop the 'Unnamed: 0' column
    # narratives = narratives.drop(columns=["Unnamed: 0"])

    # Rename the 'id' column to 'article_index'
    narratives = narratives.rename(columns={"id": "article_index"})

    return narratives


def replace_args_with_cluster_labels(
    df: pd.DataFrame, entities_directory: pd.DataFrame, cluster: str = "category"
) -> pd.DataFrame:
    """
    Replace values in 'ARG0' and 'ARG1' columns of the input DataFrame based on
    matching 'ARG0_raw' and 'ARG1_raw' values with entities in the provided
    entities_directory, and substituting them with corresponding cluster labels.

    If 'ARG0_raw' and 'ARG1_raw' do not exist in the DataFrame, they will be created
    as copies of 'ARG0' and 'ARG1' respectively.

    Parameters:
    -----------
    df : pd.DataFrame
        The main DataFrame containing the columns 'ARG0' and 'ARG1', which represent
        arguments that need to be replaced based on a lookup.

    entities_directory : pd.DataFrame
        A DataFrame containing at least two columns: 'entity' (the raw entity names)
        and the specified cluster column, which holds the cluster labels to be
        substituted in place of raw values in 'ARG0' and 'ARG1'.

    cluster : str, optional, default="category"
        The column in `entities_directory` that contains the cluster labels. This is
        the column whose values will replace the original values in 'ARG0' and 'ARG1'.

    Returns:
    --------
    pd.DataFrame
        The input DataFrame with 'ARG0' and 'ARG1' columns replaced by the corresponding
        cluster labels from the entities_directory, if a match is found. If no match is
        found, the original value is retained.
    """

    # Check if 'ARG0_raw' and 'ARG1_raw' exist; if not, create them as copies of 'ARG0' and 'ARG1'
    if "ARG0_raw" not in df.columns:
        df["ARG0_raw"] = df["ARG0"]
    if "ARG1_raw" not in df.columns:
        df["ARG1_raw"] = df["ARG1"]

    # Create a lookup dictionary from the entities_directory for efficient replacement
    entity_dict = dict(zip(entities_directory["entity"], entities_directory[cluster]))

    # Helper function to perform the replacement
    def replace_with_cluster(value, entity_dict):
        return entity_dict.get(value, value)

    # Apply the replacement function to both 'ARG0' and 'ARG1' columns
    df["ARG0"] = df["ARG0_raw"].apply(replace_with_cluster, entity_dict=entity_dict)
    df["ARG1"] = df["ARG1_raw"].apply(replace_with_cluster, entity_dict=entity_dict)

    return df


# %%
# *1. Read files
type = "news"
timestamp = "2024-11-05_19-17-41"
# 1.1 Read the svos
df = pd.read_csv(f"../output/svo_{timestamp}.csv.gz", compression="gzip")

# %%
# 1.2 Read labelled clusters
labeled_entities = pd.read_csv("../data/actor_entity_directory_v2.csv")
# 1.3 drop the duplicate entries
labeled_entities = labeled_entities.drop_duplicates(subset="entity")

# %%
# * 2. Replace the original entities with labels
df = replace_args_with_cluster_labels(df, labeled_entities, cluster="category")

# %%
# * 3. Re-run the narrative model to formulate new entity clusters
# 3.1 Prepare data
known_entities_list = labeled_entities["category"].to_list()
svos = df.apply(lambda row: {col: row[col] for col in df.columns if pd.notna(row[col])}, axis=1).tolist()

# %%
# 3.2 Fit narrative model
m = NarrativeModel(
    # clustering="kmeans",
    # PCA=False,  # dimentionality reduction
    # UMAP=False,  # dimentionality reduction
    roles_considered=["ARG0", "B-V", "ARG1"],
    roles_with_known_entities=["ARG0", "ARG1"],
    known_entities=known_entities_list,
    # assignment_to_known_entities="embeddings",
    roles_with_unknown_entities=["ARG0", "ARG1"],
    threshold=0.1,
)

m.fit(svos, progress_bar=True)

# %%
# 3.3 Extract narratives
narratives = m.predict(svos, progress_bar=True)
# %%
# 3.4 Clean the narratives
pretty_narratives = []
for n in narratives:
    pretty_narratives.append(prettify(n))
narratives_df = pd.DataFrame(narratives)


def clean_column(column):
    return column.str.split("|").str[0]


narratives_df["ARG0"] = clean_column(narratives_df["ARG0"])
narratives_df["ARG1"] = clean_column(narratives_df["ARG1"])


# %%
# * 4. Get the cluster output and observe the results

# Prepare a dictionary to hold cluster labels and their respective most frequent phrases
cluster_data = {}
optimal_model_idx = m.index_optimal_model  # Get the optimal model index
cluster_labels = m.labels_unknown_entities[optimal_model_idx].values()  # Get all cluster labels

# %% Loop through each cluster label
# Loop through each cluster label and store the phrases along with their counts in a dictionary
for label in cluster_labels:
    if label == -1:
        continue  # Skip noise or unclustered points
    try:
        # Attempt to inspect the cluster for this label
        most_common_phrases = m.inspect_cluster(label, optimal_model_idx, topn=10000)

        # Create a list of "phrase (count)" for each cluster
        cluster_data[label] = [f"{phrase} ({count})" for phrase, count in most_common_phrases]

    except KeyError as e:
        # Handle the KeyError gracefully and print which key caused the error
        print(f"KeyError for label {label} in cluster inspection: {str(e)}")

    except IndexError as e:
        # Handle IndexError which might occur when trying to access a key
        print(f"IndexError for label {label}: {str(e)}")

    except Exception as e:
        # Handle any other unexpected errors
        print(f"Unexpected error for label {label}: {str(e)}")

# Convert the dictionary into a DataFrame
cluster_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in cluster_data.items()]))

# %% Save the clusters DataFrame to an Excel file
output_path = f"../output/{type}_entity_clusters_output_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
cluster_df.to_excel(output_path, index=False)


# %%
# * Read labelled clusters after making changes
labeled_entities = pd.read_csv("../data/actor_entity_directory_v2.csv")
# drop the duplicate entries
labeled_entities = labeled_entities.drop_duplicates(subset="entity")

# %%
cluster = "category"
entity_dict = dict(zip(labeled_entities["entity"], labeled_entities[cluster]))


def replace_with_cluster(value, entity_dict):
    return entity_dict.get(value, value)


# creating new columns for relabelled entities
narratives_df["ARG0"] = narratives_df["ARG0"].apply(replace_with_cluster, entity_dict=entity_dict)
narratives_df["ARG1"] = narratives_df["ARG1"].apply(replace_with_cluster, entity_dict=entity_dict)

# %%
# * clean the df

# %%
# Drop rows where both 'ARG0' and 'ARG1' are NaN
narratives_df = narratives_df.dropna(subset=["ARG0", "ARG1"], how="all")
# %%
# Create the new 'narratives' column by concatenating 'ARG0', 'B-V', and 'ARG1' with spaces in between
narratives_df["narratives"] = (
    narratives_df["ARG0"].fillna("") + " " + narratives_df["B-V"].fillna("") + " " + narratives_df["ARG1"].fillna("")
)

# strip any leading or trailing spaces that may occur if any of the columns are empty
narratives_df["narratives"] = narratives_df["narratives"].str.strip()

# %%
# Create a new column 'svo_type' based on whether both 'ARG0' and 'ARG1' are not null
narratives_df["svo_type"] = narratives_df.apply(
    lambda row: "triplet" if pd.notna(row["ARG0"]) and pd.notna(row["ARG1"]) else "doublets", axis=1
)
# %%
# * Export narratives
# Store narratives df
narratives_df = clean_output(narratives=narratives_df)
narratives_df["svo_index"] = np.arange(len(narratives_df))
# %%
# Create a filename with the timestamp included
narratives_df.to_csv(
    f"../output/{type}_narratives_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv.gz",
    compression="gzip",
    index=False,
)

# %%
