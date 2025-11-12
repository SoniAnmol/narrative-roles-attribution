""""This script prepares the training data for annotation"""

# %%
import pandas as pd

# %%
# * methods


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


# %%
# * main
if __name__ == "__main__":
    # * Read files
    df = pd.read_csv("../output/news_narratives_2024-11-05_20-50-54.csv.gz")
    actors = pd.read_csv("../data/actor_entity_directory_v2.csv")
    sampled_data = pd.read_excel("../output/annotated_data.xlsx")
    # %%
    # * Filter actors of interest
    actors = actors[actors.keep == 1]
    actor_categories = list(actors.category.unique())
    # under_sampled_actors = [
    #     "regional government",
    #     "critical infra",
    #     "essential goods",
    #     "bad weather",
    #     "professional",
    #     "agriculture",
    #     "climate change",
    #     "environmentalist",
    #     "environment",
    #     "national government",
    # ]
    # actor_categories = under_sampled_actors
    # %%
    # * Filter articles mentioning actors of interest
    df = df[df["ARG0"].isin(actor_categories) | df["ARG1"].isin(actor_categories)]

    # %%
    # * Filter previously sampled data
    # df = df[~df["article_index"].isin(sampled_data.article_index)]
    df = df[df["sentence_index"].isin(sampled_data.sentence_index)]
    # %%
    # * Prepare sample dataset for annotation
    # sampled_data = sample_top_sentence_indices(df, top_n=20, actor_categories=actor_categories)

    # sample SVOs with same sentence index
    df = pd.concat([sampled_data, df])
    df = (
        df.groupby("sentence_index", group_keys=False)
        .apply(lambda group: group.drop_duplicates(subset=["ARG0_raw", "ARG1_raw"]))
        .reset_index(drop=True)
    )
    sampled_data = df[df["arg0_role"].isna() & df["arg1_role"].isna()]
    # %%
    # * Export sample dataset for annotation
    sampled_data.to_excel("../output/training_data_for_annotation_v3.xlsx", index=False)

# %%
