# This script process the results and prepares it for creating visualizations

# %% import libraries
from calendar import c
from math import e
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib import axis, gridspec
from matplotlib.transforms import offset_copy
import imageio
import os
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# from responses import start


# globals
ROOT = Path(__file__).resolve().parent.parent

# %% methods


def plot_top_roles_trends(
    output,
    output_clean,
    top_n=12,
    flood_date="2023-05-01",
    show_total_line=True,
    figure_export=None,
    add_stats_annotations=False
):

    # ----------------------------------------
    # Identify role columns
    # ----------------------------------------
    role_cols = [
        c for c in output_clean.columns
        if (c.endswith("-hero") or c.endswith("-villain") or c.endswith("-victim"))
        and not c.endswith("_prob")
    ]

    hero_cols_all = [c for c in role_cols if c.endswith("-hero")]
    villain_cols_all = [c for c in role_cols if c.endswith("-villain")]
    victim_cols_all = [c for c in role_cols if c.endswith("-victim")]

    # ----------------------------------------
    # Monthly aggregation
    # ----------------------------------------
    df = output_clean.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month")[role_cols].sum().reset_index()

    df_raw = output.copy()
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw["month"] = df_raw["date"].dt.to_period("M").dt.to_timestamp()

    monthly_articles = (
        df_raw.groupby("month")["doc_id"]
        .nunique()
        .reset_index()
        .rename(columns={"doc_id": "article_count"})
    )

    # Remove roles that never appear in this subset
    role_cols_nonzero = [c for c in role_cols if monthly[c].sum() > 0]

    if len(role_cols_nonzero) == 0:
        # Early exit: no roles yet in this animation frame
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No role data yet", ha="center", va="center", fontsize=20)
        fig.savefig(figure_export, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    # ----------------------------------------
    # Find global top-N roles
    # ----------------------------------------
    global_counts = monthly[role_cols].sum().sort_values(ascending=False)
    top_roles = global_counts.head(top_n).index.tolist()

    top_heroes = [c for c in top_roles if c in hero_cols_all]
    top_villains = [c for c in top_roles if c in villain_cols_all]
    top_victims = [c for c in top_roles if c in victim_cols_all]

    # ----------------------------------------
    # Normalize to percentages
    # ----------------------------------------
    monthly["global_total"] = monthly[role_cols].sum(axis=1)
    pct = monthly.copy()
    for col in role_cols:
        pct[col] = (pct[col] / pct["global_total"]) * 100

    # ----------------------------------------
    # Colors
    # ----------------------------------------
    def get_palette(n, cmap_name, minimum=0.35, maximum=1.0):
        base = cm.get_cmap(cmap_name)
        vals = np.linspace(minimum, maximum, n)
        colors = [base(v) for v in vals]
        return colors[::-1]

    hero_colors = get_palette(len(top_heroes), "Greens")
    villain_colors = get_palette(len(top_villains), "Purples")
    victim_colors = get_palette(len(top_victims), "Blues")

    # ----------------------------------------
    # Article count subplot
    # ----------------------------------------
    def plot_article_counts(ax, df_articles):
        ax.plot(
            df_articles["month"],
            df_articles["article_count"],
            color="black",
            linewidth=2.5,
            marker="o"
        )

        for x, y in zip(df_articles["month"], df_articles["article_count"]):
            ax.text(
                x, y + 1.5, str(y), fontsize=18,
                color="black", ha="center", va="bottom"
            )

        flood_dt = pd.Timestamp(flood_date)
        ax.axvline(flood_dt, color="red", linestyle="--", linewidth=1.3)
        ax.text(
            flood_dt, df_articles["article_count"].max() + 250,
            "Floods", fontsize=20, color="red", ha="center"
        )

        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_title("Article Count Per Month", fontsize=22, style="italic")

    # ----------------------------------------
    # Layered stacked subplot with smart y-limits
    # ----------------------------------------
    def plot_layered(
        ax, df, cols, colors, title,
        base=None, band=None,
        show_total_line=True,
        add_stats_annotations=False
    ):
        """
        df: pct dataframe
        cols: role columns for this band
        base: baseline array (size=months)
        band: (low, high) background highlight band
        """

        if len(cols) == 0:
            ax.set_title(title + " (no roles)")
            ax.set_ylim(0, 1)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        # baseline
        if base is None:
            base = np.zeros(len(df), dtype=float)
        else:
            base = np.asarray(base)

        # highlight band background
        if band is not None:
            ax.axhspan(band[0], band[1], facecolor="#ffffff", alpha=0.8, zorder=0)

        # Sort roles by total
        cols_sorted = df[cols].sum().sort_values(ascending=False).index.tolist()
        col_color = dict(zip(cols_sorted, colors))

        # Compute stacking
        stacked_bottom = {}
        stacked_top = {}
        cumulative = base.copy()

        for col in cols_sorted:
            bottom = cumulative.copy()
            top = bottom + df[col].values
            stacked_bottom[col] = bottom
            stacked_top[col] = top
            cumulative = top

        total_curve = cumulative

        # -------------------------------------
        # SMART Y-LIMIT LOGIC
        # -------------------------------------
        band_min = base.min()
        band_max = total_curve.max()
        padding = max((band_max - band_min) * 0.08, 0.75)

        ymin = band_min - padding
        ymax = band_max + padding

        ax.set_ylim(ymin, ymax)

        # Draw stacked layers
        layer_alpha = 0.75
        for col in cols_sorted:
            color = col_color[col]

            ax.fill_between(
                df["month"],
                stacked_bottom[col],
                stacked_top[col],
                color=color,
                alpha=layer_alpha,
                edgecolor=color,
                linewidth=1.4
            )

            ax.plot(
                df["month"],
                stacked_top[col],
                color="white",
                linewidth=1.5,
                alpha=0.9
            )

        # Total line
        if show_total_line:
            ax.plot(
                df["month"], total_curve,
                color="black", linewidth=2.4, marker="o"
            )

        # Flood line
        flood_dt = pd.Timestamp(flood_date)
        ax.axvline(flood_dt, color="red", linestyle="--", linewidth=1.3)

        # Clean axis
        for spine in ax.spines.values():
            spine.set_visible(False)

        # ----------------------------------------
        # BAND-SPECIFIC Y-TICK LABELING
        # ----------------------------------------
        yticks = np.arange(int(ymin)//10 * 10, int(ymax)//10 * 10 + 20, 10)
        labels = []

        if band is not None:
            band_low, band_high = band
        else:
            band_low, band_high = ymin, ymax

        for y in yticks:
            if band_low <= y <= band_high:
                labels.append(f"{y}%")
            else:
                labels.append("")

        ax.set_yticks(yticks)
        ax.tick_params(axis="y", color="#fff")
        ax.set_yticklabels(labels, fontsize=20, color="#444", x=0.04)

        # ----------------------------------------
        # WHITE GRIDLINES
        # ----------------------------------------
        for y in yticks:
            ax.hlines(
                y,
                df["month"].min(),
                df["month"].max(),
                colors="white",
                linestyles=":",
                linewidth=0.8,
                alpha=0.6
            )

        for x in df["month"]:
            ax.vlines(
                x,
                ymin,
                ymax,
                colors="white",
                linestyles=":",
                linewidth=0.6,
                alpha=0.5
            )

        # ----------------------------------------
        # RIGHT-SIDE LABELS — ROBUST NON-OVERLAPPING
        # ----------------------------------------
        band_mid_last = {
            col: (stacked_bottom[col][-1] + stacked_top[col][-1]) / 2
            for col in cols_sorted
        }
        order = sorted(cols_sorted, key=lambda c: band_mid_last[c], reverse=True)

        # compute desired positions
        desired_positions = [band_mid_last[col] for col in order]
        desired_positions = sorted(desired_positions, reverse=True)

        # enforce minimum spacing
        min_gap = 5
        label_positions = []

        for pos in desired_positions:
            if not label_positions:
                label_positions.append(pos)
                continue

            last = label_positions[-1]
            if last - pos < min_gap:
                pos = last - min_gap

            # keep labels within axis limits
            pos = max(pos, ymin + min_gap)

            label_positions.append(pos)

        # map back to columns
        final_positions = dict(zip(order, label_positions))

        # render labels
        last_x = df["month"].max()
        box_offset = pd.Timedelta(days=3)
        text_offset = pd.Timedelta(days=6)

        for col in order:
            color = col_color[col]
            y_target = final_positions[col]

            label = col.split("-")[0]

            if add_stats_annotations:
                s = df[col]
                label += f" (Start: {s.iloc[0]:.1f}%, End: {s.iloc[-1]:.1f}%, Avg: {s.mean():.1f}%)"

            # connector line
            y_start = band_mid_last[col]
            ax.plot(
                [last_x, last_x + box_offset],
                [y_start, y_target],
                color="#555", linewidth=1.2, alpha=0.9
            )

            # color box
            ax.add_patch(
                mpatches.Rectangle(
                    (last_x + box_offset, y_target - 0.5),
                    width=pd.Timedelta(days=2),
                    height=1,
                    facecolor=color,
                    edgecolor="#333",
                    linewidth=1.0,
                    alpha=layer_alpha,
                    transform=ax.transData,
                    clip_on=False
                )
            )

            # text label
            ax.text(
                last_x + text_offset,
                y_target,
                label,
                fontsize=20,
                color="black",
                va="center",
                ha="left"
            )

        ax.set_title(title, fontsize=22, style="italic", y=0.9, pad=2.5)

    # ----------------------------------------
    # BASELINES FOR EACH SUBPLOT
    # ----------------------------------------
    victim_total = pct[top_victims].sum(axis=1) if top_victims else pd.Series([0] * len(pct), index=pct.index)
    villain_total = pct[top_villains].sum(axis=1) if top_villains else pd.Series([0] * len(pct), index=pct.index)

    base_victims = np.zeros(len(pct))
    base_villains = victim_total.values
    base_heroes = (victim_total + villain_total).values

    # ----------------------------------------
    # Figure layout
    # ----------------------------------------
    fig = plt.figure(figsize=(20, 30), dpi=300)
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 5.25, 4.5, 4.5], hspace=0.03)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)

    # Article count subplot
    plot_article_counts(ax0, monthly_articles)

    # Final stacked plots
    plot_layered(ax1, pct, top_heroes, hero_colors, "Heroes",
                 base=base_heroes, band=(60, 100),
                 show_total_line=show_total_line,
                 add_stats_annotations=add_stats_annotations)

    plot_layered(ax2, pct, top_villains, villain_colors, "Villains",
                 base=base_villains, band=(30, 60),
                 show_total_line=show_total_line,
                 add_stats_annotations=add_stats_annotations)

    plot_layered(ax3, pct, top_victims, victim_colors, "Victims",
                 base=base_victims, band=(0, 30),
                 show_total_line=show_total_line,
                 add_stats_annotations=add_stats_annotations)

    # Shared x-axis formatting
    for ax in [ax0, ax1, ax2, ax3]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        # move x-tick labels upward
        ax.tick_params(axis="x", labelsize=20, pad=0, color="#ffffff")

    for ax in [ax0, ax1, ax2]:
        ax.tick_params(axis="x", labelbottom=False, color="#ffffff")

    ax.set_xlim(pct["month"].min(), pct["month"].max())
    ax.margins(x=0.01)

    plt.tight_layout()

    if figure_export:
        plt.savefig(
            figure_export,
            dpi=120,
            bbox_inches=None,
            transparent=True
        )

    plt.show()


def make_roles_trend_gif(
    output,
    output_clean,
    gif_path="roles_evolution.gif",
    fps=1,
    temp_dir="gif_frames",
    **plot_kwargs
):
    """
    Creates a GIF animation of the plot_top_roles_trends figure
    evolving month-by-month.
    """

    # ------------------------------------------------------------
    # Work on datetime-safe copies
    # ------------------------------------------------------------
    out_dt = output.copy()
    out_clean_dt = output_clean.copy()

    out_dt["date"] = pd.to_datetime(out_dt["date"])
    out_clean_dt["date"] = pd.to_datetime(out_clean_dt["date"])

    # ------------------------------------------------------------
    # Identify sorted unique months from the CLEAN dataframe
    # ------------------------------------------------------------
    out_clean_dt["month"] = (
        out_clean_dt["date"].dt.to_period("M").dt.to_timestamp()
    )
    months = sorted(out_clean_dt["month"].unique())

    # ------------------------------------------------------------
    # Prepare temporary folder
    # ------------------------------------------------------------
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    frame_paths = []

    # ------------------------------------------------------------
    # Generate frames month-by-month
    # ------------------------------------------------------------
    for i, m in enumerate(months):
        print(f"Rendering frame {i+1}/{len(months)} for month {m.date()}")

        # Subset both datasets up to this month (now both are datetime)
        output_sub = out_dt[out_dt["date"] <= m].copy()
        output_clean_sub = out_clean_dt[out_clean_dt["date"] <= m].copy()

        frame_file = os.path.join(temp_dir, f"frame_{i:03d}.png")

        # Call your existing plotting function
        plot_top_roles_trends(
            output=output_sub,
            output_clean=output_clean_sub,
            figure_export=frame_file,
            **plot_kwargs
        )

        # Close the figure to avoid memory buildup
        plt.close("all")
        frame_paths.append(frame_file)

    # ------------------------------------------------------------
    # Build GIF from saved frames
    # ------------------------------------------------------------
    print("Assembling GIF...")
    with imageio.get_writer(gif_path, mode="I", fps=fps) as writer:
        for fp in frame_paths:
            writer.append_data(imageio.imread(fp))

    print(f"GIF saved at: {gif_path}")

# %% main


if __name__ == "__main__":

    # read files
    # Combine predictions with original corpus
    results_dir = Path(ROOT) / "data/predictions/chunks"
    chunk_files = sorted(results_dir.glob("pred_*.parquet"))
    output = pd.concat([pd.read_parquet(cf) for cf in chunk_files], ignore_index=True)

    # remove actor-role combinations with poor F1 score
    f1_score_path = Path(ROOT) / "data/model_performance/classification_report.xlsx"
    f1_score = pd.read_excel(f1_score_path)
    drop_roles = f1_score.loc[f1_score['f1-score'] < 0.5, 'Unnamed: 0']
    output.drop(columns=drop_roles, inplace=True)

    # rename actor for clarity
    output.rename(columns={'region-hero': 'regional government-hero',
                           'region-villain': 'regional government-villain',
                           'region-victim': 'regional government-victim', }, inplace=True)

    actor_role_cols = output.columns[output.columns.str.endswith(("-hero", "-villain", "-victim"))].tolist()

    metadata_cols = ['doc_id', 'sentence', 'sentence_id', 'sentence_global_id',
                     'title', 'date', 'publisher', 'text', 'source',
                     'publisher_category', 'doc', 'sentiment_label', 'sentiment',
                     'sentence_word_count']


# %% aggregate results at sentence level by keeping first
    agg_dict = {**{c: "first" for c in metadata_cols},
                **{c: "max" for c in actor_role_cols}}

    df_sentence = output.groupby("sentence_global_id", as_index=False).agg(agg_dict)

    # remove rows with no roles
    no_role_mask = (df_sentence[actor_role_cols].sum(axis=1) == 0)
    print(f"Dropping {no_role_mask.sum()} sentences with no narrative roles")
    df_sentence = df_sentence[~no_role_mask].copy()
    print(f"Kept {len(df_sentence)} Sentences with narrative character roles")

# %% aggregate results at article level by keeping first
    metadata_cols = ['doc_id',
                     'title', 'date', 'publisher', 'text', 'source',
                     'publisher_category', 'doc', 'sentiment',
                     'sentence_word_count']
    agg_dict = {**{c: "first" for c in metadata_cols},
                **{c: "max" for c in actor_role_cols}}
    agg_dict["sentiment"] = "median"
    df_article = output.groupby("doc_id", as_index=False).agg(agg_dict)
    df_article["n_sentences"] = output.groupby("doc_id").size().values
    # remove rows with no roles
    no_role_mask = (df_article[actor_role_cols].sum(axis=1) == 0)
    print(f"Dropping {no_role_mask.sum()} articles with no narrative roles")
    df_article = df_article[~no_role_mask].copy()
    print(f"Kept {len(df_article)} Articles with narrative character roles")

    # %% Plot top role trends overtime
    # figure_export = Path(ROOT) / "figures/role_trends.png"
    # plot_top_roles_trends(df_article, df_article, top_n=14,
    #                       show_total_line=False, figure_export=figure_export,
    #                       add_stats_annotations=False)

    # figure_export = Path(ROOT) / "figures/role_trends_detailed.png"
    # plot_top_roles_trends(df_article, df_article, top_n=14,
    #                       show_total_line=False, figure_export=figure_export,
    #                       add_stats_annotations=True)

    # %% Create role trend GIF without annotation

    make_roles_trend_gif(
        df_article,       # output
        df_article,       # output_clean (or the correct clean df)
        gif_path="roles_animation.gif",
        fps=1,
        add_stats_annotations=False,
        show_total_line=False,
    )
    # %% Create role trend GIF with annotation

    make_roles_trend_gif(
        df_article,       # output
        df_article,       # output_clean (or the correct clean df)
        gif_path="roles_animation_annotated.gif",
        fps=1,
        add_stats_annotations=True,
        show_total_line=False,
    )

    # %% Plot comparative roles with survey data

    # read survey data
    survey_data_path = Path(ROOT) / "data/survey_data/survey_data.csv"
    survey_data = pd.read_csv(survey_data_path)
    # clean survey data
    # subset the dataframe by only selecting the responses that reached the end of the survey and selected 'Submit'
    survey_data = survey_data[survey_data['Q50'] == 'Submit']
    # remove responses filled during testing through preview
    survey_data = survey_data[survey_data.Status == "IP Address"]

    survey_data.dropna(subset=['Q20_0_GROUP', 'Q17_0_GROUP', 'Q12_0_GROUP'], inplace=True)
    print(f"Valid response count: {len(survey_data)}")

    # map narrative character roles form the survey questions
    actor_mapper = {'local businesses': 'business',
                    'local businesses and enterprises': 'business',
                    'residents/local population': 'people',
                    'Houses and buildings': 'people',
                    'Critical infrastructure, including roads and power lines': 'essential goods and infrastructure',
                    'local residents': 'people',
                    'farmers and agricultural companies': 'agriculture',
                    'the farmers': 'agriculture',
                    'the ngos': 'civil society',
                    'The national government': 'national government',
                    'The regional government': 'regional government',
                    'Local authorities': 'municipality',
                    'The environment, including plants and animals': 'environment'}

    for role in ['hero', 'villain', 'victim']:
        if role == 'hero':
            govt_col = 'Q21'
            role_col = 'Q20_0_GROUP'

        elif role == 'villain':
            govt_col = 'Q13'
            role_col = 'Q12_0_GROUP'

        elif role == 'victim':
            role_col = 'Q17_0_GROUP'

        survey_data[role] = [role.split(',')[0] for role in survey_data[role_col]]
        survey_data[role] = [s.strip().lower() for s in survey_data[role]]
        if role != 'victim':
            survey_data.loc[
                survey_data[role] == "the government", role] = survey_data.loc[
                survey_data[role] == "the government", govt_col]
        survey_data[role] = survey_data[role].replace(actor_mapper)
        print(survey_data[role].value_counts())
    # add victim_non_living
    survey_data['victim_non_living'] = survey_data['Q16'].replace(actor_mapper)
    print(survey_data['victim_non_living'].value_counts())

    # %% compute the role counts for newspaper corpus
    newspaper_role_counts = []
    for role in ['hero', 'villain', 'victim']:
        role_cols = output.columns[output.columns.str.endswith((f"-{role}"))].tolist()
        newspaper_role_counts.append(df_article[role_cols].sum(axis=0))
    newspaper_role_counts = pd.concat(newspaper_role_counts)

    # %% plot the dumbell chart
    # Read the role count df
    role_counts = pd.read_excel(Path(ROOT) / "data/survey_data/roles_comparisions.xlsx", sheet_name=1)

    # Colorblind-friendly palette (Wong palette, suitable for Nature Climate Change)
    COLORBLIND_COLORS = {
        0: '#D55E00',  # Vermillion
        1: '#E69F00',  # Orange
        2: '#0072B2',  # Blue
        3: '#56B4E9',  # Sky Blue
        4: '#009E73',  # Bluish Green
        5: '#F0E442',  # Yellow
        6: '#CC79A7',  # Reddish Purple
    }

    # Flag to control value annotations (True = show values, False = show x-axis instead)
    show_value_annotations = False

    fig = plt.figure(figsize=(16, 26), dpi=300)
    gs = gridspec.GridSpec(
        3, 1,
        height_ratios=[2.75, 5.5, 5],
        hspace=0.3
    )

    role_to_ax = {
        'victim': fig.add_subplot(gs[0, 0]),
        'villain': fig.add_subplot(gs[1, 0]),
        'hero': fig.add_subplot(gs[2, 0])
    }

    for role in ['victim', 'villain', 'hero']:
        ax = role_to_ax[role]
        df_role = role_counts[role_counts['role'] == role].copy()

        # Calculate percentage share for each source
        for source in df_role['source'].unique():
            total = df_role.loc[df_role['source'] == source, 'count'].sum()
            df_role.loc[df_role['source'] == source, 'pct_share'] = (
                df_role.loc[df_role['source'] == source, 'count'] / total * 100
            )

        # Pivot to get sources as columns
        df_plot = df_role.pivot(index='actor', columns='source', values='pct_share')
        df_plot = df_plot.rename(columns=str.lower)

        # Determine actor ordering based on total counts
        actor_order = (
            df_role.groupby("actor")["count"]
            .sum()
            .sort_values(ascending=True)
            .index.tolist()
        )

        df_plot = df_plot.reindex(actor_order).reset_index()
        df_plot['actor'] = df_plot['actor'].astype(str)

        # Dynamically detect available sources
        available_sources = [col for col in df_plot.columns if col != 'actor']

        # Assign colors to sources
        source_colors = {src: COLORBLIND_COLORS[i] for i, src in enumerate(available_sources)}

        # Plot data for each actor
        for idx, row in df_plot.iterrows():
            actor = row['actor']

            # Get values for all available sources
            source_values = {src: row[src] if src in row.index and not pd.isna(row[src]) else None
                           for src in available_sources}

            # Filter out None values
            valid_sources = {src: val for src, val in source_values.items() if val is not None}

            # Skip if no valid data for this actor
            if not valid_sources:
                continue

            # Plot points for each source
            for src, val in valid_sources.items():
                if src == 'newspaper':
                    ax.scatter(
                        val, actor,
                        facecolors='none',                     # hollow
                        edgecolors=source_colors[src],         # colored outline
                        s=250, zorder=2,
                        linewidths=2.5
                    )
                else:
                    ax.scatter(
                        val, actor,
                        color=source_colors[src],              # filled
                        s=250, zorder=2,
                        edgecolors='white',
                        linewidths=2
                    )

            # Draw connecting lines between consecutive points
            if len(valid_sources) >= 2:
                sorted_sources = sorted(valid_sources.items(), key=lambda x: x[1])

                for i in range(len(sorted_sources) - 1):
                    src1, val1 = sorted_sources[i]
                    src2, val2 = sorted_sources[i + 1]

                    ax.plot(
                        [val1, val2],
                        [actor, actor],
                        color='#AAAAAA', linewidth=2.5, zorder=1, alpha=0.5
                    )

            # Add value labels with alternating positions for better readability
            if show_value_annotations:
                for src_idx, (src, val) in enumerate(sorted(valid_sources.items(), key=lambda x: x[1])):
                    # Alternate between top and bottom for better spacing
                    if len(valid_sources) == 3:
                        # For 3 categories: bottom, top, bottom
                        if src_idx == 1:
                            text_offset = offset_copy(ax.transData, fig=ax.figure, x=0, y=0.12)
                            va = 'bottom'
                        else:
                            text_offset = offset_copy(ax.transData, fig=ax.figure, x=0, y=-0.12)
                            va = 'top'
                    elif len(valid_sources) == 2:
                        # For 2 categories: both bottom but with horizontal offset
                        if src_idx == 0:
                            text_offset = offset_copy(ax.transData, fig=ax.figure, x=-0.3, y=-0.12)
                        else:
                            text_offset = offset_copy(ax.transData, fig=ax.figure, x=0.3, y=-0.12)
                        va = 'top'
                    else:
                        # For 1 category: centered below
                        text_offset = offset_copy(ax.transData, fig=ax.figure, x=0, y=-0.12)
                        va = 'top'

                    ax.text(
                        val, actor, f"{val:.1f}%",
                        color='#000000', fontsize=18, ha='center', va=va,
                        transform=text_offset, fontweight='medium'
                    )

            # Add difference annotation only for large differences
            if len(valid_sources) >= 2:
                min_val = min(valid_sources.values())
                max_val = max(valid_sources.values())
                diff = max_val - min_val

                if diff >= 15:  # Increased threshold for cleaner look
                    mid_x = min_val + diff / 2
                    ax.text(
                        mid_x, actor, f"Δ{diff:.0f}%",
                        color='#555555', fontsize=14, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 edgecolor='#CCCCCC', alpha=0.8),
                        fontweight='bold'
                    )

            # Actor label on the left
            ax.text(
                -15, actor, actor,
                color='#000000', fontsize=18,
                ha='right', va='center', fontweight='medium'
            )

            # Horizontal gridline for each actor
            ax.hlines(
                actor, xmin=-12, xmax=100,
                color='#DDDDDD', linewidth=1, zorder=0, alpha=0.5
            )

        # Vertical line separating labels from plot
        ax.axvline(x=-12, color='#CCCCCC', linewidth=1.5)

        # Title
        ax.text(
            0.5, 1.08,
            f'{role.capitalize()} Role Comparison',
            transform=ax.transAxes,
            ha='center', va='bottom',
            fontsize=22, fontweight='bold'
        )

        # Set axis limits
        ax.set_xlim(-12, 100)

        # Create dynamic legend based on available sources
        legend_elements = [
    Line2D(
        [0], [0],
        marker='o',
        color='w',
        label=src.replace('_', ' ').title(),
        markerfacecolor='none' if src == 'newspaper' else source_colors[src],
        markeredgecolor=source_colors[src],
        markeredgewidth=2,
        markersize=12
    )
    for src in available_sources
]

        ax.legend(handles=legend_elements, loc='lower right', fontsize=16, frameon=True,
                 fancybox=False, shadow=False, framealpha=0.95, edgecolor='#CCCCCC')

        # Always hide y-axis ticks and labels
        ax.set_yticks([])
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        # Show x-axis when value annotations are off
        if not show_value_annotations:
            ax.set_xlabel('Percentage Share (%)', fontsize=18, fontweight='medium')
            ax.tick_params(axis='x', which='major', labelsize=16, length=6, width=1.5,
                          colors='#333333', pad=8)
            ax.spines['bottom'].set_visible(True)
            ax.spines['bottom'].set_color('#CCCCCC')
            ax.spines['bottom'].set_linewidth(1.5)
            # Set x-ticks
            ax.set_xticks(range(0, 101, 10))
            ax.set_xticklabels([f'{x}%' for x in range(0, 101, 10)])
            # Add minor gridlines on x-axis
            ax.grid(axis='x', which='major', color='#EEEEEE', linestyle='-', linewidth=1, alpha=0.7)
            # Hide x-ticks when annotations are on
            ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
        else:
            # Hide x-axis when annotations are on
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

        # Always keep top, right, left spines hidden
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        Path(ROOT) / "figures/all_roles_dumbbell_affected_unaffected.png",
        dpi=300,
        bbox_inches='tight',
        transparent=False
    )
    plt.show()

    # %% plot the dumbell chart for self and peer role attributions
    # Read the role count df
    # Expected data structure for sheet 2:
    # Columns: ['role', 'actor', 'source', 'count']
    # - role: narrative role type (hero, villain, victim)
    # - actor: entity/group (e.g., municipality, people, business, government, agriculture, etc.)
    # - source: attribution type (e.g., 'self', 'peer', 'affected_self', 'affected_peer',
    #           'unaffected_self', 'unaffected_peer')
    # - count: number of role attributions
    # The plot will dynamically detect available sources and create dumbbell connections between them.
    role_counts_peers = pd.read_excel(Path(ROOT) / "data/survey_data/roles_comparisions.xlsx", sheet_name=2)

    # Colorblind-friendly palette for self vs peer
    PEER_COLORS = {
        'self': '#0072B2',   # Blue
        'peer': '#009E73',   # Bluish Green
    }

    # Flag to control value annotations
    show_value_annotations_peer = False

    fig_peer = plt.figure(figsize=(16, 26), dpi=300)
    gs_peer = gridspec.GridSpec(
        3, 1,
        height_ratios=[2.75, 5.5, 5],
        hspace=0.3
    )

    role_to_ax_peer = {
        'victim': fig_peer.add_subplot(gs_peer[0, 0]),
        'villain': fig_peer.add_subplot(gs_peer[1, 0]),
        'hero': fig_peer.add_subplot(gs_peer[2, 0])
    }

    for role in ['victim', 'villain', 'hero']:
        ax = role_to_ax_peer[role]
        df_role = role_counts_peers[role_counts_peers['role'] == role].copy()

        # Group by actor and subjective role attribution (self vs peer)
        # Aggregate across source (affected vs unaffected)
        df_role_grouped = df_role.groupby(['actor', 'subjective role attribution'], as_index=False).agg({
            'count': 'sum'
        })

        # Calculate percentage share for each attribution type (self vs peer)
        for attr_type in df_role_grouped['subjective role attribution'].unique():
            total = df_role_grouped.loc[df_role_grouped['subjective role attribution'] == attr_type, 'count'].sum()
            df_role_grouped.loc[df_role_grouped['subjective role attribution'] == attr_type, 'pct_share'] = (
                df_role_grouped.loc[df_role_grouped['subjective role attribution'] == attr_type, 'count'] / total * 100
            )

        # Pivot to get attribution types as columns
        df_plot = df_role_grouped.pivot(index='actor', columns='subjective role attribution', values='pct_share')
        df_plot = df_plot.rename(columns=str.lower)

        # Determine actor ordering based on total counts
        actor_order = (
            df_role_grouped.groupby("actor")["count"]
            .sum()
            .sort_values(ascending=True)
            .index.tolist()
        )

        df_plot = df_plot.reindex(actor_order).reset_index()
        df_plot['actor'] = df_plot['actor'].astype(str)

        # Dynamically detect available sources
        available_sources = [col for col in df_plot.columns if col != 'actor']

        # Assign colors to sources based on mapping
        source_colors = {src: PEER_COLORS.get(src.lower(), f'#{i:02x}{i:02x}{i:02x}')
                        for i, src in enumerate(available_sources)}

        # Create label mapping for display
        source_labels = {
            'self': 'self role-attribution',
            'peer': 'subjective peer role-attribution'
        }
        source_display_labels = {
            src: source_labels.get(src.lower(), src.replace('_', ' ').title())
            for src in available_sources
        }

        # Plot data for each actor
        for idx, row in df_plot.iterrows():
            actor = row['actor']

            # Get values for all available sources
            source_values = {src: row[src] if src in row.index and not pd.isna(row[src]) else None
                           for src in available_sources}

            # Filter out None values
            valid_sources = {src: val for src, val in source_values.items() if val is not None}

            # Skip if no valid data for this actor
            if not valid_sources:
                continue

            # Plot points for each source
            for src, val in valid_sources.items():
                ax.scatter(val, actor, color=source_colors[src], s=250, zorder=2,
                         edgecolors='white', linewidths=2)

            # Draw connecting lines between consecutive points
            if len(valid_sources) >= 2:
                sorted_sources = sorted(valid_sources.items(), key=lambda x: x[1])

                for i in range(len(sorted_sources) - 1):
                    src1, val1 = sorted_sources[i]
                    src2, val2 = sorted_sources[i + 1]

                    ax.plot(
                        [val1, val2],
                        [actor, actor],
                        color='#AAAAAA', linewidth=2.5, zorder=1, alpha=0.5
                    )

            # Add value labels with alternating positions for better readability
            if show_value_annotations_peer:
                for src_idx, (src, val) in enumerate(sorted(valid_sources.items(), key=lambda x: x[1])):
                    # Alternate between top and bottom for better spacing
                    if len(valid_sources) == 2:
                        # For 2 categories: both bottom but with horizontal offset
                        if src_idx == 0:
                            text_offset = offset_copy(ax.transData, fig=fig_peer, x=-0.3, y=-0.12)
                        else:
                            text_offset = offset_copy(ax.transData, fig=fig_peer, x=0.3, y=-0.12)
                        va = 'top'
                    else:
                        # For 1 category: centered below
                        text_offset = offset_copy(ax.transData, fig=fig_peer, x=0, y=-0.12)
                        va = 'top'

                    ax.text(
                        val, actor, f"{val:.1f}%",
                        color='#000000', fontsize=18, ha='center', va=va,
                        transform=text_offset, fontweight='medium'
                    )

            # Add difference annotation only for large differences
            if len(valid_sources) >= 2:
                min_val = min(valid_sources.values())
                max_val = max(valid_sources.values())
                diff = max_val - min_val

                if diff >= 15:  # Increased threshold for cleaner look
                    mid_x = min_val + diff / 2
                    ax.text(
                        mid_x, actor, f"Δ{diff:.0f}%",
                        color='#555555', fontsize=14, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 edgecolor='#CCCCCC', alpha=0.8),
                        fontweight='bold'
                    )

            # Actor label on the left
            ax.text(
                -15, actor, actor,
                color='#000000', fontsize=18,
                ha='right', va='center', fontweight='medium'
            )

            # Horizontal gridline for each actor
            ax.hlines(
                actor, xmin=-12, xmax=100,
                color='#DDDDDD', linewidth=1, zorder=0, alpha=0.5
            )

        # Vertical line separating labels from plot
        ax.axvline(x=-12, color='#CCCCCC', linewidth=1.5)

        # Title
        ax.text(
            0.5, 1.08,
            f'{role.capitalize()} Role: Self vs Peer Attribution',
            transform=ax.transAxes,
            ha='center', va='bottom',
            fontsize=22, fontweight='bold'
        )

        # Set axis limits
        ax.set_xlim(-12, 100)

        # Create dynamic legend based on available sources
        legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                   label=source_display_labels[src],
                   markerfacecolor=source_colors[src],
                   markersize=12,
                   markeredgecolor='white',
                   markeredgewidth=2)
            for src in available_sources
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=16, frameon=True,
                 fancybox=False, shadow=False, framealpha=0.95, edgecolor='#CCCCCC')

        # Always hide y-axis ticks and labels
        ax.set_yticks([])
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        # Show x-axis when value annotations are off
        if not show_value_annotations_peer:
            ax.set_xlabel('Percentage Share (%)', fontsize=18, fontweight='medium')
            ax.tick_params(axis='x', which='major', labelsize=16, length=6, width=1.5,
                          colors='#333333', pad=8)
            ax.spines['bottom'].set_visible(True)
            ax.spines['bottom'].set_color('#CCCCCC')
            ax.spines['bottom'].set_linewidth(1.5)
            # Set x-ticks
            ax.set_xticks(range(0, 101, 10))
            ax.set_xticklabels([f'{x}%' for x in range(0, 101, 10)])
            # Add minor gridlines on x-axis
            ax.grid(axis='x', which='major', color='#EEEEEE', linestyle='-', linewidth=1, alpha=0.7)
            ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
        else:
            # Hide x-axis when annotations are on
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

        # Always keep top, right, left spines hidden
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        Path(ROOT) / "figures/all_roles_dumbbell_self_peer.png",
        dpi=300,
        bbox_inches='tight',
        transparent=False
    )
    plt.show()

    # %% Grouped Bar Chart: Self vs Peer attribution by Source (Treatment Effect)
    # This plot shows how self and peer attribution differs between flooded and non-flooded municipalities

    # Read the data from sheet 2
    df_role_treatment = pd.read_excel(
        Path(ROOT) / "data/survey_data/roles_comparisions.xlsx",
        sheet_name=2,
        engine="openpyxl"
    )

    # Define colors
    ATTRIBUTION_COLORS = {
        'Self - Flooded': '#D55E00',  # Vermillion
        'Peer - Flooded': '#E69F00',  # Orange
        'Self - Not Flooded': '#0072B2',  # Blue
        'Peer - Not Flooded': '#009E73'   # Bluish Green
    }

    # Group by source, actor, role, and subjective role attribution
    df_grouped = df_role_treatment.groupby(['source', 'actor', 'role', 'subjective role attribution']).agg({
                                           'count': 'sum'}).reset_index()

    # Calculate total counts per source and role
    totals = df_grouped.groupby(['source', 'role'])['count'].sum().reset_index()
    totals.rename(columns={'count': 'total'}, inplace=True)

    # Merge to calculate percentages
    df_grouped = df_grouped.merge(totals, on=['source', 'role'])
    df_grouped['percentage'] = (df_grouped['count'] / df_grouped['total']) * 100

    # Create a combined column for source + attribution
    df_grouped['category'] = df_grouped['source'].str.replace(
        'Flooded Municipality', 'Flooded') + ' - ' + df_grouped['subjective role attribution'].str.capitalize()

    # Pivot to get each category as a column
    df_pivot = df_grouped.pivot_table(
        index=['actor', 'role'],
        columns='category',
        values='percentage',
        fill_value=0
    ).reset_index()

    # Calculate average importance (sum of all attributions) for sorting
    for col in ['Flooded - Self', 'Flooded - Peer', 'Not Flooded - Self', 'Not Flooded - Peer']:
        if col not in df_pivot.columns:
            df_pivot[col] = 0

    df_pivot['avg_attribution'] = (df_pivot['Flooded - Self'] + df_pivot['Flooded - Peer'] +
                                   df_pivot['Not Flooded - Self'] + df_pivot['Not Flooded - Peer']) / 4

    # Create three separate plots, one for each role
    role_titles = {'victim': 'Victim', 'villain': 'Villain', 'hero': 'Hero'}

    for role in ['victim', 'villain', 'hero']:
        # Create individual figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Filter data for this role
        df_plot = df_pivot[df_pivot['role'] == role].copy()

        if df_plot.empty:
            print(f"No data for {role}")
            plt.close(fig)
            continue

        # Sort by average attribution and select top 12
        df_plot = df_plot.sort_values('avg_attribution', ascending=True).tail(12)

        # Prepare data for plotting
        actors = df_plot['actor'].values
        n_actors = len(actors)
        y_positions = np.arange(n_actors)

        bar_height = 0.2

        # Plot grouped bars
        ax.barh(y_positions - 1.5*bar_height, df_plot['Flooded - Self'], bar_height,
               label='Self - Flooded municipalities', color=ATTRIBUTION_COLORS['Self - Flooded'],
               edgecolor='white', linewidth=1)
        ax.barh(y_positions - 0.5*bar_height, df_plot['Flooded - Peer'], bar_height,
               label='Peer - Flooded municipalities', color=ATTRIBUTION_COLORS['Peer - Flooded'],
               edgecolor='white', linewidth=1)
        ax.barh(y_positions + 0.5*bar_height, df_plot['Not Flooded - Self'], bar_height,
               label='Self - Not flooded municipalities', color=ATTRIBUTION_COLORS['Self - Not Flooded'],
               edgecolor='white', linewidth=1)
        ax.barh(y_positions + 1.5*bar_height, df_plot['Not Flooded - Peer'], bar_height,
               label='Peer - Not flooded municipalities', color=ATTRIBUTION_COLORS['Peer - Not Flooded'],
               edgecolor='white', linewidth=1)

        # Add percentage labels at the end of each bar
        for i in range(n_actors):
            # Flooded - Self
            val = df_plot['Flooded - Self'].iloc[i]
            if val > 0:
                ax.text(val + 0.5, y_positions[i] - 1.5*bar_height, f'{val:.1f}%',
                       va='center', ha='left', fontsize=10, color='#333333', )

            # Flooded - Peer
            val = df_plot['Flooded - Peer'].iloc[i]
            if val > 0:
                ax.text(val + 0.5, y_positions[i] - 0.5*bar_height, f'{val:.1f}%',
                       va='center', ha='left', fontsize=10, color='#333333', )

            # Not Flooded - Self
            val = df_plot['Not Flooded - Self'].iloc[i]
            if val > 0:
                ax.text(val + 0.5, y_positions[i] + 0.5*bar_height, f'{val:.1f}%',
                       va='center', ha='left', fontsize=10, color='#333333', )

            # Not Flooded - Peer
            val = df_plot['Not Flooded - Peer'].iloc[i]
            if val > 0:
                ax.text(val + 0.5, y_positions[i] + 1.5*bar_height, f'{val:.1f}%',
                       va='center', ha='left', fontsize=10, color='#333333',)

            # Add dotted line between Flooded group and Not Flooded group
            ax.axhline(y=y_positions[i], color='#CCCCCC', linestyle=':',
                      linewidth=1.5, alpha=0.7, zorder=0)

        # Add horizontal separator lines between actor groups
        for i in range(n_actors - 1):
            line_y = (y_positions[i] + y_positions[i+1]) / 2
            ax.axhline(y=line_y, color='#DDDDDD', linestyle='-', linewidth=1, alpha=0.5, zorder=0)

        # Set actor names on y-axis
        ax.set_yticks(y_positions)
        ax.set_yticklabels(actors, fontsize=14, color='#333333')

        # Title and labels
        ax.set_title(f'{role_titles[role]} Role Attribution: Self vs Peer by Flood Status',
                    fontsize=18, fontweight='bold', pad=20, color='#2C3E50')

        # Hide x-axis completely
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax.tick_params(axis='y', which='major', labelsize=13, colors='#333333')
        ax.spines['bottom'].set_visible(False)

        # Remove gridlines
        ax.grid(False)

        # Hide other spines
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)

        # Set x-axis limits
        ax.set_xlim(0, max(df_plot[['Flooded - Self', 'Flooded - Peer',
                                     'Not Flooded - Self', 'Not Flooded - Peer']].max()) * 1.1)

        # Add legend with reversed order
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='lower right', frameon=True, fancybox=True,
                 shadow=False, fontsize=11, framealpha=0.95,
                 edgecolor='#CCCCCC', ncol=1)

        plt.tight_layout()
        plt.savefig(
            Path(ROOT) / f"figures/role_{role}_grouped_bars_treatment_effect.png",
            dpi=300,
            bbox_inches='tight',
            transparent=False
        )
        plt.show()
        plt.close(fig)

   # %%
   def plot_single_role_trend(
        output_clean,
        role="hero",
        top_n=12,
        flood_date="2023-05-01",
        show_total_line=True,
        figure_export=None
    ):
        """
        Plot a single narrative role (hero, villain, or victim).
        Matches styling of plot_top_roles_trends().
        """

        # Validate input
        if role not in ["hero", "villain", "victim"]:
            raise ValueError("role must be 'hero', 'villain', or 'victim'")

        # ---------------------------------------------------------
        # Identify relevant columns
        # ---------------------------------------------------------
        role_suffix = f"-{role}"
        role_cols = [
            c for c in output_clean.columns
            if c.endswith(role_suffix) and not c.endswith("_prob")
        ]

        if len(role_cols) == 0:
            raise ValueError(f"No columns found for role: {role}")

        # ---------------------------------------------------------
        # Prepare monthly data
        # ---------------------------------------------------------
        df = output_clean.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

        monthly = df.groupby("month")[role_cols].sum().reset_index()

        # Normalize (%)
        monthly["global_total"] = monthly[role_cols].sum(axis=1)
        pct = monthly.copy()
        for col in role_cols:
            pct[col] = (pct[col] / pct["global_total"]) * 100

        # ---------------------------------------------------------
        # Select top-N roles for this role type
        # ---------------------------------------------------------
        global_counts = monthly[role_cols].sum().sort_values(ascending=False)
        top_cols = global_counts.head(top_n).index.tolist()

        # ---------------------------------------------------------
        # Color palette
        # ---------------------------------------------------------
        def get_palette(n, cmap_name, minimum=0.35, maximum=1.0):
            base = cm.get_cmap(cmap_name)
            vals = np.linspace(minimum, maximum, n)
            colors = [base(v) for v in vals]
            return colors[::-1]

        if role == "hero":
            colors = get_palette(len(top_cols), "Greens")
        elif role == "villain":
            colors = get_palette(len(top_cols), "Purples")
        else:
            colors = get_palette(len(top_cols), "Blues")

        col_color = dict(zip(top_cols, colors))

        # ---------------------------------------------------------
        # Compute layer stacking
        # ---------------------------------------------------------
        df_plot = pct[["month"] + top_cols].copy()

        cols_sorted = df_plot[top_cols].sum().sort_values(ascending=False).index.tolist()

        stacked_bottom = {}
        stacked_top = {}

        cumulative = np.zeros(len(df_plot))
        for col in cols_sorted:
            bottom = cumulative.copy()
            top = bottom + df_plot[col].values
            stacked_bottom[col] = bottom
            stacked_top[col] = top
            cumulative = top

        row_sums = df_plot[cols_sorted].sum(axis=1)
        y_max = row_sums.max() * 1.05

        # ---------------------------------------------------------
        # FIGURE
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(18, 10), dpi=300)

        # Layered plot
        for col in cols_sorted:
            ax.fill_between(
                df_plot["month"],
                stacked_bottom[col],
                stacked_top[col],
                alpha=0.75,
                color=col_color[col],
                edgecolor=col_color[col],
                linewidth=1.4
            )
            ax.plot(
                df_plot["month"],
                stacked_top[col],
                color="white",
                linewidth=1.5,
                alpha=0.9
            )

        # Total line optionally
        if show_total_line:
            ax.plot(df_plot["month"], row_sums, color="black", linewidth=2.4, marker="o")

        # Flood vertical line
        flood_dt = pd.Timestamp(flood_date)
        ax.axvline(flood_dt, color="red", linestyle="--", linewidth=1.3)

        # ---------------------------------------------------------
        # Labels on right side (same float connector logic)
        # ---------------------------------------------------------
        band_mid_last = {
            c: (stacked_bottom[c][-1] + stacked_top[c][-1]) / 2 for c in cols_sorted
        }
        connector_order = sorted(cols_sorted, key=lambda c: band_mid_last[c], reverse=True)

        last_x = df_plot["month"].max()
        box_offset = pd.Timedelta(days=3)
        text_offset = pd.Timedelta(days=6)

        used_positions = []
        min_gap = 1.2

        for col in connector_order:
            color = col_color[col]
            y_start = band_mid_last[col]
            y_target = y_start

            for used in used_positions:
                if abs(y_target - used) < min_gap:
                    y_target = used - min_gap

            used_positions.append(y_target)

            # connector line
            ax.plot([last_x, last_x + box_offset], [y_start, y_target],
                    color="#666", linewidth=1.2, alpha=0.9)

            # color square
            ax.add_patch(
                mpatches.Rectangle(
                    (last_x + box_offset, y_target - 0.5),
                    width=pd.Timedelta(days=2),
                    height=1,
                    facecolor=color,
                    edgecolor="#555",
                    linewidth=1.0,
                    alpha=0.75,
                    transform=ax.transData,
                    clip_on=False
                )
            )

            # text label
            label = col.split("-")[0]
            ax.text(last_x + text_offset, y_target, label,
                    fontsize=16, color="black", va="center", ha="left")

        # ---------------------------------------------------------
        # Aesthetics
        # ---------------------------------------------------------
        ax.set_ylim(0, y_max)
        ax.set_title(f"{role.capitalize()} Role Trends", fontsize=24, style="italic")

        # X-axis formatting
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.tick_params(axis="x", labelsize=16)

        # Clean y-axis
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_ylabel("Percentage (%)", fontsize=18)

        plt.tight_layout()

        if figure_export:
            plt.savefig(figure_export, dpi=300, bbox_inches='tight', transparent=True)

        plt.show()

# %%
    for role in ['hero', 'villain', 'victim']:
        if role == 'hero':
            top_n = 7
        elif role == 'villain':
            top_n = 3
        else:
            top_n = 5
        plot_single_role_trend(df_article, role=role, top_n=top_n, figure_export= Path(ROOT) / f"figures/{role}_trend.png", show_total_line=False)

# %%
#
