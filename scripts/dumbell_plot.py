# %%
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# %% ---- Inputs ----
role_counts = pd.read_excel(Path(ROOT) / "data/survey_data/roles_comparisions.xlsx", sheet_name=1)

COLORBLIND_COLORS = {
    0: '#D55E00',  # Vermillion
    1: '#E69F00',  # Orange
    2: '#0072B2',  # Blue
    3: '#56B4E9',  # Sky Blue
    4: '#009E73',  # Bluish Green
    5: '#F0E442',  # Yellow
    6: '#CC79A7',  # Reddish Purple
}

show_value_annotations = False  # keep same behavior
roles = ['victim', 'villain', 'hero']

# %% ---- Figure layout (3 stacked panels with similar height ratios) ----
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=False,
    vertical_spacing=0.06,
    row_heights=[2.75, 5.5, 5],
    subplot_titles=[f"{r.capitalize()} Role Comparison" for r in roles]
)

# ---- Build each panel ----
for r_i, role in enumerate(roles, start=1):
    df_role = role_counts[role_counts['role'] == role].copy()

    # pct_share per source (same as your logic)
    df_role['source'] = df_role['source'].astype(str)
    for source in df_role['source'].unique():
        total = df_role.loc[df_role['source'] == source, 'count'].sum()
        df_role.loc[df_role['source'] == source, 'pct_share'] = (
            df_role.loc[df_role['source'] == source, 'count'] / total * 100
        )

    # pivot
    df_plot = df_role.pivot(index='actor', columns='source', values='pct_share')
    df_plot.columns = [str(c).lower() for c in df_plot.columns]
    df_plot = df_plot.reset_index()
    df_plot['actor'] = df_plot['actor'].astype(str)

    # actor ordering based on total counts (ascending)
    actor_order = (
        df_role.groupby("actor")["count"]
        .sum()
        .sort_values(ascending=True)
        .index.astype(str)
        .tolist()
    )
    df_plot = df_plot.set_index('actor').reindex(actor_order).reset_index()

    # available sources + colors
    available_sources = [c for c in df_plot.columns if c != 'actor']
    source_colors = {src: COLORBLIND_COLORS[i] for i, src in enumerate(available_sources)}

    # --- connectors + points per actor ---
    # We'll add one connector trace per actor to preserve "between consecutive points" feel.
    for _, row in df_plot.iterrows():
        actor = row['actor']
        vals = []
        for src in available_sources:
            v = row[src] if (src in row.index and pd.notna(row[src])) else None
            if v is not None:
                vals.append((src, float(v)))

        if not vals:
            continue

        # sort by value and connect consecutive (same as matplotlib)
        vals_sorted = sorted(vals, key=lambda x: x[1])

        if len(vals_sorted) >= 2:
            xs = [v for _, v in vals_sorted]
            ys = [actor] * len(xs)
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys,
                    mode="lines",
                    line=dict(color="#AAAAAA", width=3),
                    opacity=0.5,
                    hoverinfo="skip",
                    showlegend=False
                ),
                row=r_i, col=1
            )

        # Δ annotation for large differences (>= 15)
        min_val = min(v for _, v in vals_sorted)
        max_val = max(v for _, v in vals_sorted)
        diff = max_val - min_val
        if diff >= 15:
            mid_x = min_val + diff / 2
            fig.add_annotation(
                x=mid_x, y=actor,
                text=f"Δ{diff:.0f}%",
                showarrow=False,
                font=dict(color="#555555", size=14),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#CCCCCC",
                borderwidth=1,
                borderpad=4,
                xanchor="center",
                yanchor="middle",
                row=r_i, col=1
            )

        # plot each source as its own point trace (so legend works cleanly)
        for src, v in vals_sorted:
            is_newspaper = (src == "newspaper")
            fig.add_trace(
                go.Scatter(
                    x=[v],
                    y=[actor],
                    mode="markers" + ("+text" if show_value_annotations else ""),
                    marker=dict(
                        size=16,  # approximate s=250 in mpl
                        color=("rgba(0,0,0,0)" if is_newspaper else source_colors[src]),
                        line=dict(color=source_colors[src] if is_newspaper else "white", width=3),
                        symbol="circle"
                    ),
                    text=[f"{v:.1f}%" if show_value_annotations else ""],
                    textposition="top center",
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        f"Role: {role}<br>"
                        f"Source: {src.replace('_',' ').title()}<br>"
                        "Share: %{x:.2f}%<extra></extra>"
                    ),
                    name=src.replace('_', ' ').title(),
                    showlegend=False  # we'll add legend once per source below
                ),
                row=r_i, col=1
            )

    # --- legend: add one dummy trace per source to show correct marker style ---
    for src in available_sources:
        is_newspaper = (src == "newspaper")
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(
                    size=16,
                    color=("rgba(0,0,0,0)" if is_newspaper else source_colors[src]),
                    line=dict(color=source_colors[src], width=3),
                    symbol="circle"
                ),
                name=src.replace('_', ' ').title(),
                showlegend=True,
                hoverinfo="skip"
            ),
            row=r_i, col=1
        )

    # --- axis styling to mimic your mpl ---
    fig.update_xaxes(
        range=[-12, 100],
        title_text=("Percentage Share (%)" if not show_value_annotations else ""),
        tickmode="array",
        tickvals=list(range(0, 101, 10)),
        ticktext=[f"{x}%" for x in range(0, 101, 10)],
        showgrid=True,
        gridcolor="#EEEEEE",
        zeroline=False,
        row=r_i, col=1
    )
    fig.update_yaxes(
        showticklabels=False,  # you manually label actors in mpl; we keep y labels hidden for the same clean look
        showgrid=False,
        row=r_i, col=1
    )

    # Vertical separator line at x = -12
    fig.add_shape(
        type="line",
        x0=-12, x1=-12,
        y0=-0.5, y1=len(actor_order)-0.5,
        xref=f"x{r_i}" if r_i > 1 else "x",
        yref=f"y{r_i}" if r_i > 1 else "y",
        line=dict(color="#CCCCCC", width=2)
    )

    # Actor labels on the left at x=-15 (like your mpl text)
    # Plotly needs per-actor annotations:
    for actor in actor_order:
        fig.add_annotation(
            x=-15, y=actor,
            text=actor,
            showarrow=False,
            xanchor="right",
            yanchor="middle",
            font=dict(color="#000000", size=16),
            row=r_i, col=1
        )

    # Horizontal gridlines per actor (your mpl hlines)
    # Plotly "shape" line for each actor:
    for actor in actor_order:
        fig.add_shape(
            type="line",
            x0=-12, x1=100,
            y0=actor, y1=actor,
            xref=f"x{r_i}" if r_i > 1 else "x",
            yref=f"y{r_i}" if r_i > 1 else "y",
            line=dict(color="#DDDDDD", width=1),
            opacity=0.5
        )

# ---- Global layout ----
fig.update_layout(
    height=1600,   # tune as needed; HTML is scrollable anyway
    width=1100,
    plot_bgcolor="white",
    paper_bgcolor="white",
    hovermode="closest",
    legend=dict(
        x=0.99, y=0.01,
        xanchor="right",
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#CCCCCC",
        borderwidth=1,
        font=dict(size=14)
    ),
    margin=dict(l=160, r=40, t=80, b=60)
)

# Remove subplot title default spacing and style it a bit closer to yours
fig.update_annotations(font=dict(size=20))

# ---- Save HTML ----
out_html = Path(ROOT) / "figures/all_roles_dumbbell_affected_unaffected_interactive.html"
fig.write_html(out_html, include_plotlyjs="cdn")
print(f"Saved: {out_html}")
# %%
