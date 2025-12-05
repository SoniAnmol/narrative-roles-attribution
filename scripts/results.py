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
from responses import start


# globals
ROOT = Path(__file__).resolve().parent.parent

# %% methods
# def plot_top_roles_trends(output,
#     output_clean,
#     top_n=12,
#     flood_date="2023-05-01",
#     show_total_line=True,
#     figure_export=None,
#     add_stats_annotations=False
# ):

#     # Identify role columns
#     role_cols = [
#         c for c in output_clean.columns
#         if (c.endswith("-hero") or c.endswith("-villain") or c.endswith("-victim"))
#         and not c.endswith("_prob")
#     ]

#     hero_cols_all = [c for c in role_cols if c.endswith("-hero")]
#     villain_cols_all = [c for c in role_cols if c.endswith("-villain")]
#     victim_cols_all = [c for c in role_cols if c.endswith("-victim")]

#     # Monthly aggregation
#     df = output_clean.copy()
#     df["date"] = pd.to_datetime(df["date"])
#     df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
#     monthly = df.groupby("month")[role_cols].sum().reset_index()

#     df_raw = output.copy()
#     df_raw["date"] = pd.to_datetime(df_raw["date"])
#     df_raw["month"] = df_raw["date"].dt.to_period("M").dt.to_timestamp()

#     monthly_articles = (
#         df_raw.groupby("month")["doc_id"]
#         .nunique()
#         .reset_index()
#         .rename(columns={"doc_id": "article_count"})
#     )

#     # Global top-N roles
#     global_counts = monthly[role_cols].sum().sort_values(ascending=False)
#     top_roles = global_counts.head(top_n).index.tolist()

#     top_heroes = [c for c in top_roles if c in hero_cols_all]
#     top_villains = [c for c in top_roles if c in villain_cols_all]
#     top_victims = [c for c in top_roles if c in victim_cols_all]

#     # Normalize to percentages
#     monthly["global_total"] = monthly[role_cols].sum(axis=1)
#     pct = monthly.copy()
#     for col in role_cols:
#         pct[col] = (pct[col] / pct["global_total"]) * 100

#     # Colors
#     def get_palette(n, cmap_name, minimum=0.35, maximum=1.0):
#         base = cm.get_cmap(cmap_name)
#         vals = np.linspace(minimum, maximum, n)   
#         colors = [base(v) for v in vals]
#         return colors[::-1]

#     hero_colors = get_palette(len(top_heroes), "Greens")
#     villain_colors = get_palette(len(top_villains), "Purples")
#     victim_colors = get_palette(len(top_victims), "Blues")

#     # ARTICLE COUNTS
#     def plot_article_counts(ax, df_articles):

#         # Line plot
#         ax.plot(
#             df_articles["month"],
#             df_articles["article_count"],
#             color="black",
#             linewidth=2.5,
#             marker="o",
#         )

#         # Annotate each month
#         for x, y in zip(df_articles["month"], df_articles["article_count"]):
#             ax.text(
#                 x,
#                 y + 1,
#                 str(y),
#                 fontsize=16,
#                 color="black",
#                 ha="center",
#                 va="bottom"
#             )

#         # Flood lines (same as other subplots)
#         flood_dt = pd.Timestamp(flood_date)
#         ax.axvline(flood_dt, color="red", linestyle="--", linewidth=1.3)
#         ax.text(pd.Timestamp("2023-05-01"),  df_articles["article_count"].max()+100, "Floods",
#                 color="red", fontsize=16, ha="center")
        
#         # No y-axis tick labels
#         ax.set_yticks([])
#         for spine in ax.spines.values():
#             spine.set_visible(False)

#         ax.set_title("Article Count Per Month", fontsize=20, style="italic")

#     # ROLE STACKED PLOTS (unchanged except for ytick spacing fix)
#     def plot_layered(ax, df, cols, colors, title, tick_start):

#         if len(cols) == 0:
#             ax.set_title(title + " (no top roles)")
#             return

#         cols_sorted = df[cols].sum().sort_values(ascending=False).index.tolist()
#         col_color = dict(zip(cols_sorted, colors))

#         # Compute stacked areas
#         stacked_bottom = {}
#         stacked_top = {}

#         cumulative = np.zeros(len(df), dtype=float)
#         for col in cols_sorted:
#             bottom = cumulative.copy()
#             top = bottom + df[col].values
#             stacked_bottom[col] = bottom
#             stacked_top[col] = top
#             cumulative = top

#         row_sums = df[cols_sorted].sum(axis=1)
#         local_max = row_sums.max()
#         y_max = local_max * 1.05
#         ax.set_ylim(0, y_max)

#         # Stacked layers
#         layer_alpha = 0.75
#         for col in cols_sorted:
#             color = col_color[col]
#             ax.fill_between(
#                 df["month"],
#                 stacked_bottom[col],
#                 stacked_top[col],
#                 alpha=layer_alpha,
#                 color=color,
#                 edgecolor=color,
#                 linewidth=1.4
#             )   
#             ax.plot(
#                 df["month"],
#                 stacked_top[col],
#                 color="white",
#                 linewidth=1.5,
#                 alpha=0.9,
#             )


#         # Total line
#         if show_total_line:
#             ax.plot(df["month"], row_sums, color="black", linewidth=2.4, marker="o")

#         # Flood lines
#         flood_dt = pd.Timestamp(flood_date)
#         ax.axvline(flood_dt, color="red", linestyle="--", linewidth=1.3)
#         # ax.text(pd.Timestamp("2023-05-01"), y_max * 0.99, "Floods",
#                 # color="red", fontsize=16, ha="center")

#         # Clean y-axis
#         for spine in ax.spines.values():
#             spine.set_visible(False)

#         # Custom y-ticks
#         yticks = np.arange(0, y_max, 10)
#         labels_full = [f"{tick_start + i * 10}%" for i in range(len(yticks))]
#         labels_sparse = [""] * len(labels_full)
#         labels_sparse[:4] = labels_full[:4]

#         ax.set_yticks(yticks)
#         ax.set_yticklabels(labels_sparse, fontsize=16, color="#555")
#         ax.tick_params(axis="y", pad=2)
#         for tick in ax.get_yticklabels():
#             tick.set_x(0.04)

#         # White gridlines
#         for y in yticks:
#             ax.hlines(y, df["month"].min(), df["month"].max(),
#                       colors="white", linestyles=":", linewidth=0.7, alpha=0.4)

#         ax.grid(which="major", color="#ffffff", alpha=0.98, linestyle=':')

#         # Right-side labels
#         band_mid_last = {}
#         for col in cols_sorted:
#             band_mid_last[col] = (
#                 stacked_bottom[col][-1] + stacked_top[col][-1]
#             ) / 2

#         connector_order = sorted(cols_sorted,
#                                  key=lambda c: band_mid_last[c],
#                                  reverse=True)

#         last_x = df["month"].max()
#         box_offset = pd.Timedelta(days=3)
#         text_offset = pd.Timedelta(days=6)

#         used_positions = []
#         min_gap = 1.2

#         for col in connector_order:
#             color = col_color[col]

#             y_start = band_mid_last[col]
#             y_target = y_start

#             for used in used_positions:
#                 if abs(y_target - used) < min_gap:
#                     y_target = used - min_gap

#             used_positions.append(y_target)
#             text = col.split('-')[0]

#             if add_stats_annotations:
#                 series = df[col]
#                 start_val = series.iloc[0]
#                 end_val   = series.iloc[-1]
#                 average_val = series.mean()
#                 text += f" (Start: {start_val:.1f}%, End: {end_val:.1f}%, Avg: {average_val:.1f}%)"

#             ax.plot([last_x, last_x + box_offset], [y_start, y_target],
#                     color="#666", linewidth=1.2, alpha=0.9)

#             square_size = 1
#             ax.add_patch(
#                 mpatches.Rectangle(
#                     (last_x + box_offset, y_target - square_size/2),
#                     width=pd.Timedelta(days=2), # pyright: ignore[reportArgumentType]
#                     height=square_size,
#                     facecolor=color,
#                     edgecolor="#555",
#                     linewidth=1.0,
#                     alpha=layer_alpha,
#                     transform=ax.transData,
#                     clip_on=False
#                 )
#             )


#             ax.text(last_x + text_offset, y_target, text,
#                     fontsize=16, color="black", va="center", ha="left")

#         ax.set_title(title, fontsize=20, pad=2.5, style="italic", y=0.95)

#     # Final Figure 
#     fig = plt.figure(figsize=(30, 24), dpi=300)
#     gs = gridspec.GridSpec(
#         4, 1,
#         height_ratios=[0.3, 1, 1, 1],
#         hspace=0.03
#     )

#     ax0 = fig.add_subplot(gs[0])

#     # Then others can share x with ax0
#     ax1 = fig.add_subplot(gs[1], sharex=ax0)
#     ax2 = fig.add_subplot(gs[2], sharex=ax0)
#     ax3 = fig.add_subplot(gs[3], sharex=ax0)

#     axes = [ax0, ax1, ax2, ax3]

#     # First subplot: article count
#     plot_article_counts(ax0, monthly_articles)

#     # Stacked narrative roles
#     plot_layered(ax1, pct, top_heroes, hero_colors,    "Heroes",   tick_start=60)
#     plot_layered(ax2, pct, top_villains, villain_colors, "Villains", tick_start=30)
#     plot_layered(ax3, pct, top_victims,  victim_colors,  "Victims",  tick_start=0)

#     for ax in axes:
#         ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
#         ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
#         ax.tick_params(axis="x", labelsize=16)
#     for ax in axes[:-1]:
#         ax.tick_params(axis="x", labelbottom=False)

#     plt.tight_layout()
#     if figure_export:
#         plt.savefig(figure_export, dpi=300, bbox_inches='tight',
#         transparent=True)
#     plt.show()


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
            ax.set_ylim(0, 100)
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
    victim_total = pct[top_victims].sum(axis=1) if len(top_victims) else pd.Series(0, index=pct.index)
    villain_total = pct[top_villains].sum(axis=1) if len(top_villains) else pd.Series(0, index=pct.index)

    base_victims = np.zeros(len(pct))
    base_villains = victim_total.values
    base_heroes  = (victim_total + villain_total).values

    # ----------------------------------------
    # Figure layout
    # ----------------------------------------
    fig = plt.figure(figsize=(30, 20), dpi=300)
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

    plt.tight_layout()

    if figure_export:
        plt.savefig(
            figure_export,
            dpi=300,
            bbox_inches="tight",
            transparent=True
        )

    plt.show()



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
    drop_roles = f1_score.loc[f1_score['f1-score']<0.5, 'Unnamed: 0']
    output.drop(columns=drop_roles, inplace=True)

    # rename actor for clarity
    output.rename(columns={'region-hero':'regional government-hero',
                           'region-villain':'regional government-villain',
                           'region-victim':'regional government-victim',}, inplace=True)

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
    figure_export = Path(ROOT) / "figures/role_trends.png"
    plot_top_roles_trends(df_article, df_article, top_n=14, 
                          show_total_line=False, figure_export=figure_export,
                          add_stats_annotations=False)
    
    figure_export = Path(ROOT) / "figures/role_trends_detailed.png"
    plot_top_roles_trends(df_article, df_article, top_n=14, 
                          show_total_line=False, figure_export=figure_export,
                          add_stats_annotations=True)

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
    actor_mapper = {'local businesses':'business',
                    'local businesses and enterprises':'business',
                    'residents/local population':'people',
                    'Houses and buildings':'people',
                    'Critical infrastructure, including roads and power lines':'essential goods and infrastructure',
                    'local residents':'people',
                    'farmers and agricultural companies':'agriculture',
                    'the farmers':'agriculture',
                    'the ngos':'civil society',
                    'The national government':'national government',
                    'The regional government':'regional government',
                    'Local authorities':'municipality',
                    'The environment, including plants and animals':'environment'}

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

    #%% compute the role counts for newspaper corpus
    newspaper_role_counts = []
    for role in ['hero', 'villain', 'victim']:
        role_cols = output.columns[output.columns.str.endswith((f"-{role}"))].tolist()
        newspaper_role_counts.append(df_article[role_cols].sum(axis=0))
    newspaper_role_counts = pd.concat(newspaper_role_counts)    

    #%% plot the dumbell chart
    # Read the role count df
    role_counts = pd.read_excel(Path(ROOT)/ "data/survey_data/roles_comparisions.xlsx")

    fig = plt.figure(figsize=(12, 24), dpi=300)
    gs = gridspec.GridSpec(
        3, 1,
        height_ratios=[2.75, 5.5, 5],
        hspace=0.25
    )

    role_to_ax = {
        'victim': fig.add_subplot(gs[0, 0]),
        'villain': fig.add_subplot(gs[1, 0]),
        'hero': fig.add_subplot(gs[2, 0])
    }
    for role in ['victim', 'villain', 'hero']:
        ax = role_to_ax[role]
        df_role = role_counts[role_counts['role'] == role].copy()
        
        for source in df_role['source'].unique():
            total = df_role.loc[df_role['source'] == source, 'count'].sum()
            df_role.loc[df_role['source'] == source, 'pct_share'] = (
                df_role.loc[df_role['source'] == source, 'count'] / total * 100
            )

        df_plot = df_role.pivot(index='actor', columns='source', values='pct_share')

        df_plot = df_plot.rename(columns=str.lower)

        actor_order = (
            df_role.groupby("actor")["count"]
            .sum()
            .sort_values(ascending=True)
            .index.tolist()
        )

        # reindex df_plot to keep ALL actors
        df_plot = df_plot.reindex(actor_order)

        df_plot = df_plot.reset_index()
        df_plot['actor'] = df_plot['actor'].astype(str)

        # check which columns actually exist
        has_survey = 'survey' in df_plot.columns
        has_newspaper = 'newspaper' in df_plot.columns

        for _, row in df_plot.iterrows():

            actor = row['actor']
            survey_val = row['survey'] if has_survey else np.nan
            news_val   = row['newspaper'] if has_newspaper else np.nan

            # Skip completely empty rows
            if pd.isna(survey_val) and pd.isna(news_val):
                continue

            # Skip actors that only have newspaper data
            if pd.isna(survey_val):
                continue

            if not pd.isna(survey_val) and not pd.isna(news_val):

                # dumbbell line
                ax.plot(
                    [survey_val, news_val],
                    [actor, actor],
                    color='gray', linewidth=2, zorder=1
                )

                # dots
                ax.scatter(survey_val, actor, color='blue', s=200, zorder=2)
                ax.scatter(news_val, actor, color='orange', s=200, zorder=2)

                # offsets
                if survey_val > news_val:
                    newspaper_text_offset = offset_copy(
                        ax.transData, fig=ax.figure, x=-0.5, y=-0.075
                    )
                    survey_text_offset = offset_copy(
                        ax.transData, fig=ax.figure, x=0.5, y=-0.075
                    )
                else:
                    newspaper_text_offset = offset_copy(
                        ax.transData, fig=ax.figure, x=0.5, y=-0.075
                    )
                    survey_text_offset = offset_copy(
                        ax.transData, fig=ax.figure, x=-0.5, y=-0.075
                    )

                # labels
                ax.text(
                    survey_val, actor, f"{survey_val:.1f}%",
                    color='#222222', fontsize=16, ha='center', va='bottom',
                    transform=survey_text_offset
                )
                ax.text(
                    news_val, actor, f"{news_val:.1f}%",
                    color='#222222', fontsize=16, ha='center', va='bottom',
                    transform=newspaper_text_offset
                )

                # diff annotation
                diff_x = abs(survey_val - news_val)
                if diff_x >= 10:
                    mid_x = min(survey_val, news_val) + diff_x / 2
                    ax.text(
                        mid_x, actor, f"{diff_x:.1f}%",
                        color='#222222', fontsize=10, ha='center', va='bottom',
                        transform=offset_copy(ax.transData, fig=ax.figure, y=0.05)
                    )


            elif not pd.isna(survey_val):
                ax.scatter(survey_val, actor, color='blue', s=200, zorder=2)
                ax.text(
                    survey_val, actor, f"{survey_val:.1f}%",
                    color='#222222', fontsize=16, ha='center', va='bottom',
                    transform=offset_copy(ax.transData, fig=ax.figure, x=0.5, y=-0.075)
                )


            ax.text(
                -12, actor,
                actor,
                color='#222222', fontsize=16,
                ha='right', va='center_baseline'
            )

            ax.hlines(
                actor, xmin=-9, xmax=95,
                color='#818589', linewidth=0.8, zorder=0,
                transform=offset_copy(ax.transData, fig=ax.figure, y=-0.5)
            )

        ax.axvline(x=-9, color='#dddddd', linewidth=0.8)

        ax.text(
            0.5, 1.1,
            f'{role.capitalize()} Role Comparison',
            transform=ax.transAxes,
            ha='center', va='bottom',
            fontsize=18, fontweight='bold'
        )

        # ax.set_xlabel('Percentage Share (%)', fontsize=14)
        ax.set_xlim(-9, 95)

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Survey',
                markerfacecolor='blue', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Newspaper',
                markerfacecolor='orange', markersize=10)
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=14)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(
        Path(ROOT) / "figures/all_roles_dumbbell.png",
        dpi=300,
        bbox_inches='tight',
        transparent=True
    )
    plt.show()


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

#%%
    for role in ['hero', 'villain', 'victim']:
        if role == 'hero':
            top_n =7
        elif role == 'villain':
            top_n =3
        else:
            top_n =5
        plot_single_role_trend(df_article, role=role, top_n=top_n,figure_export= Path(ROOT) / f"figures/{role}_trend.png", show_total_line=False)

# %%
