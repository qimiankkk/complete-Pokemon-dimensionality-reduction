"""
Visualization layer: all Plotly figure builders for the dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

TYPE_COLORS = {
    "fire": "#FF4136", "water": "#0074D9", "grass": "#2ECC40",
    "electric": "#FFDC00", "psychic": "#F012BE", "ice": "#7FDBFF",
    "dragon": "#7B68EE", "dark": "#333333", "fairy": "#FFB3DE",
    "normal": "#AAAAAA", "fighting": "#C0392B", "flying": "#89CFF0",
    "poison": "#9B59B6", "ground": "#D4A843", "rock": "#9B8651",
    "bug": "#8BC34A", "ghost": "#5C6BC0", "steel": "#78909C",
    "None": "#DDDDDD",
}

# Cluster color palette (for up to 20 clusters)
CLUSTER_COLORS = px.colors.qualitative.Plotly + px.colors.qualitative.D3


def build_latent_scatter(
    df_display: pd.DataFrame,
    color_by: str,
    spotlight: str | None,
    manifest: dict,
    show_hulls: bool,
    hull_data: dict | None,
) -> go.Figure:
    """
    Build the main WebGL latent space scatter plot.
    df_display must have columns: emb_x, emb_y, name, type1, type2,
    pokedex_number, hp, attack, cluster, and all needed for coloring.
    """
    fig = go.Figure()

    # Determine color mapping
    if color_by == "type1":
        categories = df_display["type1"].unique()
        color_map = TYPE_COLORS
    elif color_by == "cluster":
        categories = sorted(df_display["cluster"].unique())
        color_map = {c: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, c in enumerate(categories)}
    elif color_by == "generation":
        categories = sorted(df_display["generation"].unique())
        gen_colors = px.colors.qualitative.Set2
        color_map = {g: gen_colors[i % len(gen_colors)] for i, g in enumerate(categories)}
    elif color_by == "is_legendary":
        categories = [0, 1]
        color_map = {0: "#AAAAAA", 1: "#FFD700"}
    else:
        categories = df_display[color_by].unique()
        color_map = {c: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, c in enumerate(categories)}

    # Add hull traces first (behind points)
    if show_hulls and hull_data:
        for cid, vertices in hull_data.items():
            hull_color = color_map.get(cid, "#CCCCCC") if color_by == "cluster" else "#CCCCCC"
            fig.add_trace(go.Scatter(
                x=vertices[:, 0],
                y=vertices[:, 1],
                fill="toself",
                mode="none",
                fillcolor=hull_color,
                opacity=0.15,
                name=f"Cluster {cid} hull",
                showlegend=False,
                hoverinfo="skip",
            ))

    # Add scatter traces (one per category for legend + spotlight)
    for cat in categories:
        mask = df_display[color_by] == cat
        subset = df_display[mask]

        if spotlight is not None and str(cat) != str(spotlight):
            opacity = 0.1
            marker_size = 6
        else:
            opacity = 0.85
            marker_size = 9 if spotlight is not None else 7

        # Build customdata for hover
        customdata = []
        for _, row in subset.iterrows():
            dex = str(int(row["pokedex_number"]))
            sprite_url = manifest.get(dex, {}).get("tooltip", "")
            customdata.append([
                row["name"],
                sprite_url,
                row["type1"],
                int(row["hp"]),
                int(row["attack"]),
            ])

        color = color_map.get(cat, "#AAAAAA")

        fig.add_trace(go.Scattergl(
            x=subset["emb_x"].values,
            y=subset["emb_y"].values,
            mode="markers",
            marker=dict(
                size=marker_size,
                color=color,
                opacity=opacity,
                line=dict(width=0.5, color="white"),
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "<img src='%{customdata[1]}' width='72'/><br>"
                "Type: %{customdata[2]}<br>"
                "HP %{customdata[3]} · ATK %{customdata[4]}"
                "<extra></extra>"
            ),
            name=str(cat),
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,1)",
        xaxis=dict(showgrid=False, zeroline=False, title="Latent Dim 1"),
        yaxis=dict(showgrid=False, zeroline=False, title="Latent Dim 2"),
        legend=dict(
            title=color_by.replace("_", " ").title(),
            font=dict(size=10),
            itemsizing="constant",
        ),
        margin=dict(l=40, r=40, t=30, b=40),
        height=600,
    )

    return fig


def build_stat_radar(
    pokemon_row: pd.Series,
    cluster_mean: pd.Series | None = None,
) -> go.Figure:
    """
    Hexagonal radar with the clicked Pokémon and optionally the cluster average.
    """
    stats = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
    labels = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]

    values = [pokemon_row[s] for s in stats]
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        name=pokemon_row["name"],
        fillcolor="rgba(99,110,250,0.3)",
        line=dict(color="rgb(99,110,250)", width=2),
    ))

    if cluster_mean is not None:
        cm_values = [cluster_mean.get(s, 0) for s in stats]
        cm_values_closed = cm_values + [cm_values[0]]
        fig.add_trace(go.Scatterpolar(
            r=cm_values_closed,
            theta=labels_closed,
            fill=None,
            name="Cluster Avg",
            line=dict(color="rgba(255,165,0,0.7)", width=2, dash="dash"),
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(values) * 1.2]),
            bgcolor="rgba(0,0,0,0)",
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        margin=dict(l=60, r=60, t=30, b=30),
        height=350,
    )

    return fig


def build_type_advantage_heatmap(pokemon_row: pd.Series) -> go.Figure:
    """
    18-cell horizontal heatmap showing against_? damage multipliers.
    """
    from data import TYPES

    type_labels = [t.capitalize() for t in TYPES]
    values = [pokemon_row.get(f"against_{t}", 1.0) for t in TYPES]

    # Custom color scale: 0x=blue, 0.5x=light blue, 1x=white, 2x=orange, 4x=red
    colorscale = [
        [0.0, "#0074D9"],      # 0x
        [0.125, "#7FDBFF"],    # 0.5x
        [0.25, "#FFFFFF"],     # 1x
        [0.5, "#FF851B"],      # 2x
        [1.0, "#FF4136"],      # 4x
    ]

    fig = go.Figure(data=go.Heatmap(
        z=[values],
        x=type_labels,
        y=["Damage"],
        colorscale=colorscale,
        zmin=0,
        zmax=4,
        text=[[f"{v}x" for v in values]],
        texttemplate="%{text}",
        textfont=dict(size=11),
        showscale=False,
        hovertemplate="<b>%{x}</b>: %{z}x<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=100,
        margin=dict(l=60, r=20, t=10, b=30),
        xaxis=dict(side="bottom", tickangle=-45),
        yaxis=dict(visible=False),
    )

    return fig


def build_knn_gallery_html(neighbor_rows: pd.DataFrame, manifest: dict) -> str:
    """
    Build an HTML flex row of Pokémon cards for the KNN gallery.
    Each card: sprite (card tier, 96px) + name + type badge.
    """
    cards = []
    for _, row in neighbor_rows.iterrows():
        dex = str(int(row["pokedex_number"]))
        sprite = manifest.get(dex, {}).get("card", "")
        type_color = TYPE_COLORS.get(row["type1"], "#AAA")
        cards.append(f"""
            <div style="text-align:center;width:110px;flex-shrink:0">
              <img src="{sprite}" width="96" style="border-radius:8px"
                   onerror="this.src='{manifest.get(dex, {}).get('tooltip', '')}'"/>
              <div style="font-weight:bold;font-size:12px;color:#fff;margin-top:4px">
                {row['name']}
              </div>
              <span style="background:{type_color};color:#fff;
                           padding:2px 6px;border-radius:4px;font-size:10px">
                {row['type1']}
              </span>
            </div>
        """)
    return f"""
        <div style="display:flex;gap:12px;overflow-x:auto;padding:8px;
                    background:rgba(20,20,30,0.5);border-radius:8px">
          {''.join(cards)}
        </div>
    """


def build_histogram(
    df: pd.DataFrame,
    col: str,
    color_by: str | None = None,
) -> go.Figure:
    """1D distribution with optional color grouping."""
    if color_by and color_by in df.columns:
        color_map = TYPE_COLORS if color_by == "type1" else None
        fig = px.histogram(
            df, x=col, color=color_by,
            color_discrete_map=color_map,
            barmode="overlay", opacity=0.7,
            template="plotly_dark",
        )
    else:
        fig = px.histogram(
            df, x=col, opacity=0.7,
            template="plotly_dark",
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,1)",
        margin=dict(l=40, r=20, t=30, b=40),
        height=400,
    )
    return fig


def build_2d_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    trendline: bool = False,
) -> go.Figure:
    """2D correlation scatter."""
    color_map = TYPE_COLORS if color == "type1" else None
    fig = px.scatter(
        df, x=x, y=y, color=color,
        color_discrete_map=color_map,
        render_mode="webgl",
        trendline="ols" if trendline else None,
        template="plotly_dark",
        hover_data=["name"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,1)",
        margin=dict(l=40, r=20, t=30, b=40),
        height=450,
    )
    return fig


def build_correlation_matrix(
    df: pd.DataFrame,
    cols: list[str],
) -> go.Figure:
    """Plotly heatmap of Pearson correlations."""
    corr = df[cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b>: %{z:.2f}<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        height=500,
        margin=dict(l=80, r=20, t=30, b=80),
    )
    return fig


def build_type_distribution(df: pd.DataFrame, type_col: str = "type1") -> go.Figure:
    """Horizontal bar chart of type counts, colored with TYPE_COLORS."""
    counts = df[type_col].value_counts().sort_values()
    colors = [TYPE_COLORS.get(t, "#AAA") for t in counts.index]

    fig = go.Figure(go.Bar(
        x=counts.values,
        y=counts.index,
        orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b>: %{x} Pokémon<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,1)",
        margin=dict(l=80, r=20, t=30, b=40),
        height=450,
        xaxis_title="Count",
        yaxis_title="",
    )
    return fig


def build_generation_breakdown(
    df: pd.DataFrame,
    stacked: bool = True,
) -> go.Figure:
    """Grouped/stacked bar chart: each generation's type composition."""
    cross = pd.crosstab(df["generation"], df["type1"])
    fig = go.Figure()

    for type_name in cross.columns:
        fig.add_trace(go.Bar(
            name=type_name,
            x=cross.index,
            y=cross[type_name],
            marker_color=TYPE_COLORS.get(type_name, "#AAA"),
        ))

    fig.update_layout(
        barmode="stack" if stacked else "group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,1)",
        margin=dict(l=40, r=20, t=30, b=40),
        height=450,
        xaxis_title="Generation",
        yaxis_title="Count",
        legend=dict(font=dict(size=9)),
    )
    return fig


def build_legendary_violin(df: pd.DataFrame) -> go.Figure:
    """Side-by-side violin plots: legendary vs non-legendary for each base stat."""
    stats = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
    df_melt = df[stats + ["is_legendary"]].melt(
        id_vars=["is_legendary"],
        value_vars=stats,
        var_name="stat",
        value_name="value",
    )
    df_melt["legendary"] = df_melt["is_legendary"].map({0: "Regular", 1: "Legendary"})

    fig = px.violin(
        df_melt, x="stat", y="value", color="legendary",
        color_discrete_map={"Regular": "#AAAAAA", "Legendary": "#FFD700"},
        template="plotly_dark",
        box=True,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,1)",
        margin=dict(l=40, r=20, t=30, b=40),
        height=400,
        xaxis_title="",
        yaxis_title="Value",
    )
    return fig
