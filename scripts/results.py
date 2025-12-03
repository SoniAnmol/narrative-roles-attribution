# This script process the results and prepares it for creating visualizations

# %% import libraries
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


# globals
ROOT = Path(__file__).resolve().parent.parent

# %% methods
def plot_top_roles_trends(
    output_clean,
    top_n=12,
    flood_date="2023-05-01",
    show_total_line=True,
    figure_export=None
):

    # Identify role columns
    role_cols = [
        c for c in output_clean.columns
        if (c.endswith("-hero") or c.endswith("-villain") or c.endswith("-victim"))
        and not c.endswith("_prob")
    ]

    hero_cols_all = [c for c in role_cols if c.endswith("-hero")]
    villain_cols_all = [c for c in role_cols if c.endswith("-villain")]
    victim_cols_all = [c for c in role_cols if c.endswith("-victim")]

    # Monthly aggregation
    df = output_clean.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month")[role_cols].sum().reset_index()

    monthly_articles = (
        df.groupby("month")["doc_id"]
        .nunique()
        .reset_index()
        .rename(columns={"doc_id": "article_count"})
    )

    # Global top-N roles
    global_counts = monthly[role_cols].sum().sort_values(ascending=False)
    top_roles = global_counts.head(top_n).index.tolist()

    top_heroes = [c for c in top_roles if c in hero_cols_all]
    top_villains = [c for c in top_roles if c in villain_cols_all]
    top_victims = [c for c in top_roles if c in victim_cols_all]

    # Normalize to percentages
    monthly["global_total"] = monthly[role_cols].sum(axis=1)
    pct = monthly.copy()
    for col in role_cols:
        pct[col] = (pct[col] / pct["global_total"]) * 100

    # Colors
    def get_palette(n, cmap_name, minimum=0.35, maximum=1.0):
        base = cm.get_cmap(cmap_name)
        vals = np.linspace(minimum, maximum, n)   
        colors = [base(v) for v in vals]
        return colors[::-1]

    hero_colors = get_palette(len(top_heroes), "Greens")
    villain_colors = get_palette(len(top_villains), "Purples")
    victim_colors = get_palette(len(top_victims), "Blues")

    # ARTICLE COUNTS
    def plot_article_counts(ax, df_articles):

        # Line plot
        ax.plot(
            df_articles["month"],
            df_articles["article_count"],
            color="black",
            linewidth=2.5,
            marker="o",
        )

        # Annotate each month
        for x, y in zip(df_articles["month"], df_articles["article_count"]):
            ax.text(
                x,
                y + 1,
                str(y),
                fontsize=16,
                color="black",
                ha="center",
                va="bottom"
            )

        # Flood lines (same as other subplots)
        flood_dt = pd.Timestamp(flood_date)
        ax.axvline(flood_dt, color="red", linestyle="--", linewidth=1.3)
        ax.text(pd.Timestamp("2023-05-01"),  df_articles["article_count"].max()+100, "Floods",
                color="red", fontsize=16, ha="center")
        
        # No y-axis tick labels
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_title("Article Count Per Month", fontsize=20, style="italic")

    # ROLE STACKED PLOTS (unchanged except for ytick spacing fix)
    def plot_layered(ax, df, cols, colors, title, tick_start):

        if len(cols) == 0:
            ax.set_title(title + " (no top roles)")
            return

        cols_sorted = df[cols].sum().sort_values(ascending=False).index.tolist()
        col_color = dict(zip(cols_sorted, colors))

        # Compute stacked areas
        stacked_bottom = {}
        stacked_top = {}

        cumulative = np.zeros(len(df), dtype=float)
        for col in cols_sorted:
            bottom = cumulative.copy()
            top = bottom + df[col].values
            stacked_bottom[col] = bottom
            stacked_top[col] = top
            cumulative = top

        row_sums = df[cols_sorted].sum(axis=1)
        local_max = row_sums.max()
        y_max = local_max * 1.05
        ax.set_ylim(0, y_max)

        # Stacked layers
        layer_alpha = 0.75
        for col in cols_sorted:
            color = col_color[col]
            ax.fill_between(
                df["month"],
                stacked_bottom[col],
                stacked_top[col],
                alpha=layer_alpha,
                color=color,
                edgecolor=color,
                linewidth=1.4
            )   
            ax.plot(
                df["month"],
                stacked_top[col],
                color="white",
                linewidth=1.5,
                alpha=0.9,
            )


        # Total line
        if show_total_line:
            ax.plot(df["month"], row_sums, color="black", linewidth=2.4, marker="o")

        # Flood lines
        flood_dt = pd.Timestamp(flood_date)
        ax.axvline(flood_dt, color="red", linestyle="--", linewidth=1.3)
        # ax.text(pd.Timestamp("2023-05-01"), y_max * 0.99, "Floods",
                # color="red", fontsize=16, ha="center")

        # Clean y-axis
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Custom y-ticks
        yticks = np.arange(0, y_max, 10)
        labels_full = [f"{tick_start + i * 10}%" for i in range(len(yticks))]
        labels_sparse = [""] * len(labels_full)
        labels_sparse[:4] = labels_full[:4]

        ax.set_yticks(yticks)
        ax.set_yticklabels(labels_sparse, fontsize=16, color="#555")
        ax.tick_params(axis="y", pad=2)
        for tick in ax.get_yticklabels():
            tick.set_x(0.04)

        # White gridlines
        for y in yticks:
            ax.hlines(y, df["month"].min(), df["month"].max(),
                      colors="white", linestyles=":", linewidth=0.7, alpha=0.4)

        ax.grid(which="major", color="#ffffff", alpha=0.98, linestyle=':')

        # Right-side labels
        band_mid_last = {}
        for col in cols_sorted:
            band_mid_last[col] = (
                stacked_bottom[col][-1] + stacked_top[col][-1]
            ) / 2

        connector_order = sorted(cols_sorted,
                                 key=lambda c: band_mid_last[c],
                                 reverse=True)

        last_x = df["month"].max()
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
            text = col.split('-')[0]
            ax.plot([last_x, last_x + box_offset], [y_start, y_target],
                    color="#666", linewidth=1.2, alpha=0.9)

            square_size = 1
            ax.add_patch(
                mpatches.Rectangle(
                    (last_x + box_offset, y_target - square_size/2),
                    width=pd.Timedelta(days=2), # pyright: ignore[reportArgumentType]
                    height=square_size,
                    facecolor=color,
                    edgecolor="#555",
                    linewidth=1.0,
                    alpha=layer_alpha,
                    transform=ax.transData,
                    clip_on=False
                )
            )

            ax.text(last_x + text_offset, y_target, text,
                    fontsize=16, color="black", va="center", ha="left")

        ax.set_title(title, fontsize=20, pad=2.5, style="italic", y=0.95)

    # Final Figure 
    fig = plt.figure(figsize=(30, 24), dpi=300)
    gs = gridspec.GridSpec(
        4, 1,
        height_ratios=[0.3, 1, 1, 1],
        hspace=0.03
    )

    ax0 = fig.add_subplot(gs[0])

    # Then others can share x with ax0
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)

    axes = [ax0, ax1, ax2, ax3]

    # First subplot: article count
    plot_article_counts(ax0, monthly_articles)

    # Stacked narrative roles
    plot_layered(ax1, pct, top_heroes, hero_colors,    "Heroes",   tick_start=60)
    plot_layered(ax2, pct, top_villains, villain_colors, "Villains", tick_start=30)
    plot_layered(ax3, pct, top_victims,  victim_colors,  "Victims",  tick_start=0)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.tick_params(axis="x", labelsize=16)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)

    plt.tight_layout()
    if figure_export:
        plt.savefig(figure_export, dpi=300)
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
                           'region-victim':'regional government-victim'}, inplace=True)

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
    plot_top_roles_trends(df_article, top_n=14, show_total_line=False, figure_export=figure_export)

    # %% TODO Plot comparative roles with survey data

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

#%% create dumbbell plots for each role

fig = plt.figure(figsize=(16, 18), dpi=300)
gs = gridspec.GridSpec(
    3, 1,
    height_ratios=[4, 2, 1.5],   # adjust these numbers as you like
    hspace=0.25
)

role_to_ax = {
    'hero': fig.add_subplot(gs[0, 0]),
    'victim': fig.add_subplot(gs[1, 0]),
    'villain': fig.add_subplot(gs[2, 0])
}

# ---------------------------------------------------------
# Your entire plotting loop (unchanged)
# ---------------------------------------------------------
for role in ['hero', 'victim', 'villain']:

    ax = role_to_ax[role]  # <---- use subplot

    df_role = role_counts[role_counts['role']==role].copy()

    # compute pct_share per source
    for source in df_role['source'].unique():
        total = df_role.loc[df_role['source']==source, 'count'].sum()
        df_role.loc[df_role['source']==source, 'pct_share'] = (
            df_role.loc[df_role['source']==source, 'count'] / total * 100
        )

    # SORT BY NEWSPAPER BEFORE PIVOT
    df_newspaper_sorted = (
        df_role[df_role['source']=="newspaper"]
        .sort_values("count", ascending=True)
    )

    actor_order = df_newspaper_sorted["actor"].tolist()

    # pivot
    df_plot = (
        df_role.pivot(index='actor', columns='source', values='pct_share')
    )

    df_plot = df_plot.rename(columns=str.lower)

    actor_order_valid = [a for a in actor_order if a in df_plot.index]
    df_plot = df_plot.loc[actor_order_valid]

    df_plot = df_plot.reset_index()
    df_plot['actor'] = df_plot['actor'].astype(str)

    # --------------- per-row plotting (unchanged) ---------------
    for _, row in df_plot.iterrows():

        if pd.isna(row['survey']) or pd.isna(row['newspaper']):
            continue

        ax.plot([row['survey'], row['newspaper']], [row['actor'], row['actor']],
                color='gray', linewidth=2, zorder=1)

        ax.scatter(row['survey'], row['actor'], color='blue', s=200, zorder=2)
        ax.scatter(row['newspaper'], row['actor'], color='orange', s=200, zorder=2)

        if row['survey'] > row['newspaper']:
            newspaper_text_offset = offset_copy(ax.transData, fig=ax.figure, x=-0.5, y=-0.07)
            survey_text_offset   = offset_copy(ax.transData, fig=ax.figure, x=0.5, y=-0.07)
        else:
            newspaper_text_offset = offset_copy(ax.transData, fig=ax.figure, x=0.5, y=-0.07)
            survey_text_offset   = offset_copy(ax.transData, fig=ax.figure, x=-0.5, y=-0.07)

        ax.text(row['survey'], row['actor'], f"{row['survey']:.1f}%",
                color='#222222', fontsize=14, ha='center', va='bottom',
                transform=survey_text_offset)

        ax.text(row['newspaper'], row['actor'], f"{row['newspaper']:.1f}%",
                color='#222222', fontsize=14, ha='center', va='bottom',
                transform=newspaper_text_offset)

        diff_x = abs(row['survey'] - row['newspaper'])
        if diff_x >= 10:
            mid_x = min(row['survey'], row['newspaper']) + diff_x / 2
            ax.text(mid_x, row['actor'], f"{diff_x:.1f}%",
                    color='#222222', fontsize=10, ha='center', va='bottom',
                    transform=offset_copy(ax.transData, fig=ax.figure, y=0.05))

        ax.text(-8, row['actor'], row['actor'], color='#222222', fontsize=16,
                ha='right', va='center_baseline')

        ax.hlines(row['actor'], xmin=-7.5, xmax=95, color='#818589',
                  linewidth=0.8, zorder=0,
                  transform=offset_copy(ax.transData, fig=ax.figure, y=-0.5))

    ax.axvline(x=-7.5, color='#dddddd', linewidth=0.8)
    ax.text(
        0.5, 1.1,                           # x=center, y=5% above plot
        f'{role.capitalize()} Role Comparison',
        transform=ax.transAxes,              # use Axes coordinates (0–1)
        ha='center', va='bottom',
        fontsize=18, fontweight='bold'
    )    
    ax.set_xlabel('Percentage Share (%)', fontsize=14)
    ax.set_xlim(-7.5, 95)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Survey',
               markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Newspaper',
               markerfacecolor='orange', markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axis('off')

# final save
plt.tight_layout()
plt.savefig(Path(ROOT) / "figures/all_roles_dumbbell.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
