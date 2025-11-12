""""This script contains methods for creating visualizations"""

# %%
# *                               Import libraries                               #

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from collections import defaultdict, Counter
import networkx as nx
import pandas as pd
from itertools import combinations
import plotly.graph_objects as go

# %%
# *                                   Methods                                   #


def filter_data(df, filter):
    if filter is None:
        return df
    else:
        return df[df["svo_type"] == filter].reset_index()


def calculate_log_odds_ratio(df, sentiment_column="sentiment", filter=None):
    """
    Calculates the log odds ratio of sentiment being positive vs negative for the entire DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing sentiment scores.
    sentiment_column : str, optional
        The name of the column containing sentiment scores. Default is 'sentiment'.
    filter : dict, optional
        A dictionary of filtering conditions to apply to the DataFrame before calculating the log odds ratio.

    Returns:
    --------
    float
        The log odds ratio (negative vs positive) for the entire DataFrame.
    """

    # Apply filtering if specified
    df = filter_data(df, filter)
    # Filter out neutral sentiments (sentiment == 0)
    df = df[df[sentiment_column] != 0]

    # Calculate the number of positive and negative sentiments
    positive_count = (df[sentiment_column] > 0).sum()
    negative_count = (df[sentiment_column] < 0).sum()

    # Apply Laplace smoothing (add 0.5 to avoid division by zero)
    odds_positive = (positive_count + 0.5) / (negative_count + 0.5)

    # Calculate the log odds ratio
    log_odds_ratio = np.log(odds_positive)

    return log_odds_ratio


def find_narratives(df, ARG0=None, ARG1=None, operator="&", filter=None):
    """
    Filters the DataFrame based on the values in ARG0 and ARG1 lists.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing 'ARG0' and 'ARG1' columns.
    ARG0 : list, optional
        The list of values to filter for in the 'ARG0' column. Default is None.
    ARG1 : list, optional
        The list of values to filter for in the 'ARG1' column. Default is None.
    operator : str, optional
        The logical operator to use for combining conditions. Can be '&' (and) or '|' (or). Default is '&'.

    Returns:
    --------
    pd.DataFrame
        A DataFrame filtered based on the conditions provided for 'ARG0' and 'ARG1'.
    """
    df = filter_data(df, filter)
    # Initialize the slicing condition
    condition = None

    # Check if ARG0 is provided, and filter for ARG0 values
    if ARG0 is not None:
        condition = df["ARG0"].isin(ARG0)

    # Check if ARG1 is provided, and filter for ARG1 values
    if ARG1 is not None:
        if condition is None:
            condition = df["ARG1"].isin(ARG1)
        else:
            # Apply the operator specified (& or |)
            if operator == "&":
                condition &= df["ARG1"].isin(ARG1)
            elif operator == "|":
                condition |= df["ARG1"].isin(ARG1)

    # If no conditions are provided, return the original DataFrame
    if condition is None:
        return df

    # Apply the filter condition to the DataFrame
    sliced_df = df[condition]

    return sliced_df


def plot_narratives_sentiment(df, narrative_col="narratives", sentiment_col="sentiment", top_n=5, filter=None):
    """
    Plots top narratives with sentiment values above 0 and those below 0 on a flat y-axis.
    Negative sentiment narratives are shown as orange triangles and positive sentiment narratives
    as green hexagons. Each narrative is annotated above its corresponding sentiment point.

    Parameters:
    df (DataFrame): The input DataFrame containing narratives and sentiment values.
    narrative_col (str): The name of the column containing narrative text.
    sentiment_col (str): The name of the column containing sentiment values.
    top_n (int): The number of top narratives to select from each sentiment group (positive and negative).
    """
    df = filter_data(df, filter)

    # Group by narrative, count occurrences, and calculate median sentiment
    narrative_stats = (
        df.groupby(narrative_col)
        .agg(
            count=(narrative_col, "size"),  # Count occurrences of each narrative
            median_sentiment=(sentiment_col, "median"),  # Calculate median sentiment for each narrative
        )
        .reset_index()
    )

    # Separate the narratives into two groups: sentiment > 0 and sentiment < 0
    positive_narratives = narrative_stats[narrative_stats["median_sentiment"] > 0]
    negative_narratives = narrative_stats[narrative_stats["median_sentiment"] < 0]

    # Sort by count in descending order and get the top N narratives from both groups
    top_positive_narratives = positive_narratives.sort_values(by="count", ascending=False).head(top_n)
    top_negative_narratives = negative_narratives.sort_values(by="count", ascending=False).head(top_n)

    # Combine the two sets of top narratives for plotting
    df_top = pd.concat([top_positive_narratives, top_negative_narratives])

    # Create the figure and axis
    plt.figure(figsize=(12, 6))

    # Plot positive sentiment narratives as green hexagons
    pos_mask = df_top["median_sentiment"] > 0
    plt.scatter(
        df_top.loc[pos_mask, "median_sentiment"],
        np.zeros(sum(pos_mask)),
        c="green",
        marker="h",
        s=150,
        zorder=2,
        label="Positive Sentiment",
    )

    # Plot negative sentiment narratives as orange triangles
    neg_mask = df_top["median_sentiment"] < 0
    plt.scatter(
        df_top.loc[neg_mask, "median_sentiment"],
        np.zeros(sum(neg_mask)),
        c="orange",
        marker="v",
        s=150,
        zorder=2,
        label="Negative Sentiment",
    )

    # Annotate each narrative above the corresponding sentiment point with staggered y-coordinates
    y_offsets = np.linspace(0.02, 0.04, len(df_top))  # Generate staggered y-coordinates
    for i, row in df_top.iterrows():
        y_offset = y_offsets[i % len(y_offsets)] * (-1 if i % 2 == 0 else 1)  # Alternate y-offset direction
        plt.text(row["median_sentiment"], y_offset, row[narrative_col], rotation=90, ha="right", fontsize=10, zorder=3)
        # Draw a vertical line from the x-axis to the narrative label
        plt.plot([row["median_sentiment"], row["median_sentiment"]], [0, y_offset], "gray", zorder=1)

    # Add some padding to the x-axis
    plt.xlim(df_top["median_sentiment"].min() - 0.2, df_top["median_sentiment"].max() + 0.2)
    plt.ylim(-0.05, 0.1)

    # Remove y-axis ticks and labels for a cleaner look
    plt.gca().get_yaxis().set_visible(False)

    plt.grid(visible=True, axis="x")
    # Set labels and title
    plt.xlabel("Sentiment")
    plt.title("Top Narratives by Positive and Negative Sentiment")

    # Add a legend to differentiate between positive and negative sentiment
    plt.legend()

    # Display the plot
    plt.show()


def top_n_cooccurring_with_narrative(df, target_narrative, top_n=10, filter=None):
    df = filter_data(df, filter)
    # Group narratives by sentence_index
    grouped = df.groupby("sentence_index")["narratives"].apply(list)

    # Find all narrative pairs for each sentence that include the target narrative
    cooccurring_narratives = []
    for narratives in grouped:
        if target_narrative in narratives:
            # Add all other narratives that co-occur with the target narrative
            cooccurring_narratives.extend([n for n in narratives if n != target_narrative])

    # Count the frequency of each co-occurring narrative
    narrative_counts = Counter(cooccurring_narratives)

    # Get the top N most frequent co-occurring narratives
    top_n_narratives = narrative_counts.most_common(top_n)

    # Convert the result to a DataFrame for easier readability
    result_df = pd.DataFrame(top_n_narratives, columns=["Narrative", "Frequency"])

    return result_df


def get_top_narratives_table(
    df, top_n=10, excel_filename="../output/top_narratives.xlsx", filter=None, plot=False, **kwargs
):
    """
    Returns a table of top narratives with ARG0, ARG1, B-V, and two example sentences for each narrative,
    includes a count of each narrative, and exports the table to an Excel file.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the narratives, ARG0, ARG1, B-V, and sentences.
    top_n : int, optional
        The number of top narratives to retrieve. Default is 10.
    excel_filename : str, optional
        The filename to which the resulting table will be exported. Default is 'top_narratives.xlsx'.
    plot : bool, optional
        If True, plot the count of top_n narratives. Default is False.
    log_scale : bool, optional (in kwargs)
        If True, plot the log(count) instead of count. Default is False.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the top narratives with two example sentences, ARG0, ARG1, B-V, and a narrative count for each.
    """
    log_scale = kwargs.get("log_scale", False)

    # Apply filter to data if a filter is provided
    df = filter_data(df, filter)

    # Get the top N most frequent narratives with their counts
    narrative_counts = df["narratives"].value_counts().head(top_n)

    # Get the top N most frequent narratives
    top_narratives = narrative_counts.index

    # Filter the DataFrame to include only the top N narratives
    filtered_df = df[df["narratives"].isin(top_narratives)]

    # Group by 'narratives' and pick 5 example sentences for each narrative
    result_df = filtered_df.groupby("narratives").head(5)

    # Add narrative count column
    result_df["narrative_count"] = result_df["narratives"].map(narrative_counts)

    # Select only the columns needed: 'narratives', 'ARG0', 'ARG1', 'B-V', 'sentence', and 'narrative_count'
    result_df = result_df[["narratives", "ARG0", "ARG1", "B-V", "sentence", "narrative_count"]]

    # Reset index for better readability
    result_df = result_df.reset_index(drop=True)

    # Export the DataFrame to an Excel file if filename is provided
    if excel_filename is not None:
        result_df.to_excel(excel_filename, index=False)
        print(f"Table exported to {excel_filename}")

    # If plot is True, generate the plot of top N narratives
    if plot:
        plt.figure(figsize=(10, 6))
        counts = narrative_counts.values

        if log_scale:
            counts = np.log1p(counts)  # Log scale (log(1+x) to handle zero values)

        plt.bar(narrative_counts.index, counts, color="skyblue")
        plt.xlabel("Narratives")
        plt.ylabel("Log Count" if log_scale else "Count")
        plt.title(f"Top {top_n} Narratives")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    return result_df


def plot_network_graph(df, title, filter=None):

    df = filter_data(df, filter)

    if len(df) == 0:
        print(f"not enough data to plot {title}")
        return None

    # Create a directed graph
    G = nx.DiGraph()

    # Create a Counter to keep track of edge frequencies and node frequencies (only for source nodes)
    edge_counter = Counter()
    source_node_counter = Counter()  # Only for source nodes

    # Add edges from df and count edge and source node occurrences
    for _, row in df.iterrows():
        edge = (row["ARG0"], row["ARG1"])
        G.add_edge(row["ARG0"], row["ARG1"], label=row["B-V"], publisher=row["publisher_name"])
        edge_counter[edge] += 1
        source_node_counter[row["ARG0"]] += 1  # Only count occurrences of source nodes

    # Total number of source node mentions to calculate percentages
    total_source_mentions = sum(source_node_counter.values())

    # Calculate percentage of mentions for each source node
    source_node_percentages = {
        node: (count / total_source_mentions) * 100 for node, count in source_node_counter.items()
    }

    # Get positions for the nodes in the graph
    pos = nx.spring_layout(G)

    # Find the maximum edge count for normalization
    max_edge_count = max(edge_counter.values())

    # Create a color map based on the frequency of edges
    cmap = plt.get_cmap("Blues")  # Using plt.get_cmap() instead of cm.get_cmap()

    # Assign edge colors based on frequency (normalized to [0, 1])
    edge_colors = [cmap(edge_counter[(u, v)] / max_edge_count) for u, v in G.edges()]

    # Normalize node size based on source node frequency (set 500 size for destination nodes)
    max_node_count = max(source_node_counter.values())
    node_sizes = [
        (source_node_counter.get(node, 1) / max_node_count * 1000) if node in source_node_counter else 5000
        for node in G.nodes()
    ]

    # Create node colors: Source nodes as one color, destination nodes as another color
    node_colors = ["lightblue" if node in source_node_counter else "orange" for node in G.nodes()]

    # Plot the graph
    plt.figure(figsize=(20, 20))

    # Adjust plot limits to avoid cutting off labels
    plt.margins(x=0.2, y=0.2)

    # Draw nodes with sizes proportional to their frequency (and different colors for source and destination nodes)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)

    # Modify source node labels to include the percentage of mentions (ignore destination node percentages)
    node_labels = {}
    for node in G.nodes():
        if node in source_node_counter:
            node_labels[node] = f"{node} ({source_node_percentages[node]:.1f}%)"
        else:
            node_labels[node] = node  # No percentage for destination nodes

    # Draw labels for nodes
    for node, label in node_labels.items():
        # Draw source node labels above the node with a white background
        if node in source_node_counter:
            plt.text(
                pos[node][0],
                pos[node][1] + 0.15,
                label,
                fontsize=10,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )
        # Draw destination node labels inside the node
        else:
            plt.text(
                pos[node][0],
                pos[node][1],
                label,
                fontsize=10,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )

    # Draw edges with varying colors based on their frequency
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrowstyle="->", arrowsize=20, edge_cmap=cmap)

    # Draw edge labels (using 'B-V' column for label)
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="green")

    # Show plot
    plt.title(title)
    plt.show()


def plot_narrative_evolution(df, top_n=10, log_scale=False, filter=None):
    """
    Plots the evolution of top N narratives over time (monthly counts).

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the 'narratives' and 'date' columns.
    top_n : int, optional
        The number of top narratives to plot. Default is 10.
    log_scale : bool, optional (in kwargs)
        If True, plot the log(count) instead of count. Default is False.

    Returns:
    --------
    None. Displays a plot showing the evolution of the top N narratives.
    """
    df = filter_data(df, filter)
    df = df[df.date.notna()]
    # Convert the 'date' column to datetime
    df["date"] = pd.to_datetime(df["date"], format="mixed")

    # Extract the year and month from the 'date' column
    df["month"] = df["date"].dt.to_period("M")

    # Get the top N most frequent narratives overall
    narrative_counts = df["narratives"].value_counts().head(top_n)
    top_narratives = narrative_counts.index

    # Filter the DataFrame to include only the top N narratives
    df_top_n = df[df["narratives"].isin(top_narratives)]

    # Group by 'month' and 'narratives' and count occurrences
    monthly_counts = df_top_n.groupby(["month", "narratives"]).size().unstack(fill_value=0)

    # Plotting the results
    plt.figure(figsize=(10, 6))

    if log_scale:
        monthly_counts = np.log1p(monthly_counts)  # Log scale (log(1+x) to avoid issues with 0)

    # Plot each narrative's monthly counts
    for narrative in top_narratives:
        plt.plot(monthly_counts.index.astype(str), monthly_counts[narrative], marker="o", label=narrative)

    plt.xlabel("Month")
    plt.ylabel("Log Count" if log_scale else "Count")
    plt.title(f"Evolution of Top {top_n} Narratives Over Time")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Narratives", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


# %%
# *                                    Main                                     #

if __name__ == "__main__":

    # ! Set file paths
    file_path = "../output/"
    output = "../figures/"

    # %%
    # * Read files
    df = pd.read_csv(f"{file_path}news_narratives_2024-11-05_20-50-54.csv.gz", compression="gzip")
    df = df.dropna(subset=["B-V"])
    # %%
    labeled_entities = pd.read_csv("../data/actor_entity_directory_v2.csv")
    # %%
    top_n = get_top_narratives_table(df, filter="triplet", plot=True, log_scale=False)

    # %%
    plot_narrative_evolution(df, top_n=5, filter="triplet", log_scale=True)

    # %%
    sliced_df = find_narratives(
        df,
        ARG0=["environmentalist", "climate change", "extreme event", "damage"],
        ARG1=["environmentalist", "climate change", "extreme event", "damage"],
        operator="&",
        filter="triplet",
    )

    # %%
    plot_narrative_evolution(df=sliced_df, top_n=5, filter="triplet", log_scale=False)

    # %%
    plot_narratives_sentiment(df=sliced_df, top_n=3)

    # %%
    actors = [
        "climate change",
        "political actors",
        "national government",
        "local government",
        "agriculture",
        "EU",
        "river monitoring",
        "emergency service",
    ]

    all_actors = labeled_entities.category.value_counts().index.to_list()

    log_odds_positive = {}
    for actor in all_actors:
        sliced_df = find_narratives(df, ARG0=[actor], ARG1=[actor], operator="|", filter=None)
        log_odds_positive[actor] = calculate_log_odds_ratio(df=sliced_df)
        df_ratio = pd.DataFrame(list(log_odds_positive.items()), columns=["Actors", "Log Odds Ratio"])

    # %%
    # df_ratio.to_excel("../output/log_odds_ratio.xlsx", index=False)
