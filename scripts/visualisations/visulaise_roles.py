""""This script creates visualizations of character roles"""

# %%
# * import libraries
from matplotlib import legend
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np


# %%
# * methods
def plot_predicted_character_role_over_time(
    df,
    date_col="date",
    role_col="predicted_character_role",
    output_file="../figures/character_roles_over_time.png",
    top_n=10,
    start_date="2023-04-01",
    end_date="2024-04-01",
):
    """
    Plots the frequency of the top N predicted character roles over time as a stacked area plot,
    within a specified date range.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        date_col (str): The name of the column containing the date values.
        role_col (str): The name of the column containing the predicted character roles.
        output_file (str): The filename to save the plot.
        top_n (int): Number of top predicted character roles to include in the plot.
        start_date (str): The start date for the plot (format: 'YYYY-MM-DD'). Default is '2023-04-01'.
        end_date (str): The end date for the plot (format: 'YYYY-MM-DD'). Default is '2024-04-01'.

    Returns:
        None
    """
    # Step 1: Convert the 'date' column to datetime format
    df[date_col] = pd.to_datetime(df[date_col], format="mixed", errors="coerce")

    # Step 2: Filter data based on the date range
    mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
    filtered_df = df[mask]

    # Step 3: Group by month and predicted_character_role
    filtered_df.loc[:, "month"] = filtered_df[date_col].dt.to_period("M")
    role_counts = filtered_df.groupby(["month", role_col]).size().reset_index(name="count")

    # Step 4: Calculate total counts per month and normalize for frequencies
    total_counts_per_month = role_counts.groupby("month")["count"].transform("sum")
    role_counts["frequency"] = (role_counts["count"] / total_counts_per_month) * 100

    # Step 5: Filter top N frequent roles across the filtered dataset
    top_roles = role_counts.groupby(role_col)["count"].sum().nlargest(top_n).index
    role_counts = role_counts[role_counts[role_col].isin(top_roles)]

    # Step 6: Pivot the table for plotting
    pivot_df = role_counts.pivot(index="month", columns=role_col, values="frequency").fillna(0)

    # Step 7: Compute start, end, and average statistics for each role
    legend_labels = []
    for role in pivot_df.columns:
        start_value = pivot_df[role].iloc[0]
        end_value = pivot_df[role].iloc[-1]
        avg_value = pivot_df[role].mean()
        legend_labels.append(f"{role} (start: {start_value:.1f}%, avg: {avg_value:.1f}%, end: {end_value:.1f}%)")

    # Step 8: Plot the area chart
    plt.figure(figsize=(16, 10))
    pivot_df.plot(kind="area", stacked=True, colormap="tab20", figsize=(16, 10))

    # Formatting the plot
    plt.title(f"Frequency of Predicted Character-Roles Over Time", fontsize=16)
    plt.xlabel("Date (Months)", fontsize=12)
    plt.ylabel("Share of Each Character Role (%)", fontsize=12)

    # Add custom legend
    plt.legend(
        legend_labels,
        title="Character Roles",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=12,  # Adjust the font size
        title_fontsize=14,  # Adjust the title font size
        labelspacing=1.2,
    )
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file, dpi=300)
    plt.show()


def plot_cooccurring_character_roles(
    df, role_col="predicted_character_role", group_col="sentence_index", output_file="cooccurrence_grid.png", top_n=10
):
    """
    Plots a grid of co-occurring character roles based on a specified grouping column
    (e.g., sentence_index or article_index), with a custom legend for frequency categories.

    Args:
        df (pd.DataFrame): The input DataFrame.
        role_col (str): The name of the column containing character roles.
        group_col (str): The column used to group and determine co-occurrences (e.g., sentence_index or article_index).
        output_file (str): The filename to save the plot.
        top_n (int): Number of top character roles to consider for the plot.

    Returns:
        None
    """
    # Step 1: Group by the specified group_col and collect all unique roles per group
    grouped_roles = df.groupby(group_col)[role_col].apply(lambda x: list(set(x)))

    # Step 2: Create a co-occurrence matrix
    role_counts = df[role_col].value_counts().nlargest(top_n)
    top_roles = role_counts.index
    unique_roles = sorted(top_roles)

    cooccurrence_matrix = pd.DataFrame(0, index=unique_roles, columns=unique_roles)

    for roles in grouped_roles:
        filtered_roles = [r for r in roles if r in top_roles]
        for role1 in filtered_roles:
            for role2 in filtered_roles:
                cooccurrence_matrix.loc[role1, role2] += 1

    # Step 3: Normalize the co-occurrence matrix by rows (convert to percentage)
    cooccurrence_percentage = cooccurrence_matrix.div(cooccurrence_matrix.sum(axis=1), axis=0) * 100

    # Step 4: Define custom colormap based on frequency levels
    norm = plt.Normalize(vmin=0, vmax=100)

    # Step 5: Plot the grid
    plt.figure(figsize=(16, 12))
    sns.heatmap(
        cooccurrence_percentage,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        norm=norm,
        linewidths=0.5,
        square=True,
        cbar=False,
    )

    # Formatting the plot
    plt.title(f"Co-occurrence Frequency of Predicted Character Roles", fontsize=16)
    plt.xlabel("Character Roles", fontsize=12)
    plt.ylabel("Character Roles", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file, dpi=300)
    plt.show()


# %%
# main
if __name__ == "__main__":
    # * read files
    df = pd.read_csv("../../output/predicted_results.csv.gz", compression="gzip")
    performance = pd.read_excel("../../output/prediction_performance.xlsx")
    actor_directory = pd.read_csv("../../data/actor_entity_directory_v2.csv")

    # %%
    # * Prepare actor directory
    actor_directory = actor_directory[actor_directory.keep == 1].reset_index(drop=True)
    actor_directory = dict(zip(actor_directory["category"], actor_directory["actor"]))
    actor_directory["municipalities"] = "municipalities"
    actor_directory["region"] = "regional government"
    actor_directory["bad weather"] = "extreme event"

    # %%
    # * Filter for character roles with acceptable f1 score
    f1_threshold = None
    if f1_threshold:
        performance = performance[performance.f1_score >= f1_threshold]
        df = df[df.predicted_character_role.isin(performance["class"])].reset_index(drop=True)

    # %%
    # * Evaluate the labelling of actors
    roles_to_evaluate = [
        "political actors-hero",
        "political actors-villain",
        "political actors-victim",
        "local government-hero",
        "local government-villain",
        "local government-victim",
        "national government-hero",
        "national government-villain",
        "national government-victim",
        "regional government-hero",
        "regional government-villain",
        "regional government-victim",
    ]
    actors_to_evaluate = [
        "local government",
        "national government",
        "regional government",
        "political actors",
        "municipalities",
        "region",
    ]
    for role in roles_to_evaluate:
        subset_df = df[df["predicted_character_role"] == role]
        try:
            for index, row in tqdm(subset_df.iterrows()):
                if row["ARG0"] in actors_to_evaluate:
                    # assign correct role
                    correct_role = str(f"{actor_directory[row['ARG0']]}-{role.split('-')[-1]}")
                    df.loc[index, "predicted_character_role"] = correct_role

                elif row["ARG1"] in actors_to_evaluate:
                    # assign correct role
                    correct_role = str(f"{actor_directory[row['ARG1']]}-{role.split('-')[-1]}")
                    df.loc[index, "predicted_character_role"] = correct_role

        except KeyError as e:
            print(f"KeyError: {e} in row {index}")
        except AttributeError as e:
            print(f"AttributeError: {e} in row {index}")
        except Exception as e:
            print(f"Unexpected error: {e} in row {index}")

    # %%
    # * Remove uncorrect roles
    df.loc[df.predicted_character_role == "flood-hero", "predicted_character_role"] = np.NaN

    # %%
    # * plot_predicted_character_role_over_time
    plot_predicted_character_role_over_time(df, top_n=10, output_file="../../figures/character_roles_over_time_10.png")



    # %%
    # * subset df for narrative roles
    narrative_roles = ["hero", "villain", "victim"]

    df = df[df["predicted_character_role"].notna()]

    for role in narrative_roles:
        subset_df = df[df["predicted_character_role"].str.endswith(f"-{role}")]
        plot_predicted_character_role_over_time(
            subset_df, top_n=10, output_file=f"../../figures/character_roles_over_time_{role}.png"
        )
        subset_df["count"] = 1  # Creates a column with a constant value
        avg_count = subset_df.groupby(["sentence_index", "predicted_character_role"])["count"].mean().reset_index()
        avg_count = avg_count.predicted_character_role.value_counts()
        print(avg_count)

    # %%

# %%
