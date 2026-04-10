"""
Stlite WebAssembly ZIP packager for offline deployment.

Exports the trained model's embeddings, cluster labels, and sprite URLs
into a self-contained stlite WebAssembly app.
"""

import json
import zipfile
import os

import pandas as pd


STLITE_INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PokéDex Latent Space</title>
  <link
    rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/@stlite/mountable@0.63.0/build/stlite.css"
  />
</head>
<body>
  <div id="root"></div>
  <script src="https://cdn.jsdelivr.net/npm/@stlite/mountable@0.63.0/build/stlite.js"></script>
  <script>
    stlite.mount({
      requirements: ["plotly", "pandas"],
      entrypoint: "app_export.py",
      files: {
        "app_export.py": { url: "./app_export.py" },
        "export_data.csv": { url: "./export_data.csv" },
        "manifest.json": { url: "./manifest.json" },
      },
    }, document.getElementById("root"));
  </script>
</body>
</html>
"""

APP_EXPORT_PY = '''"""Standalone exported Pokémon Latent Space viewer (no PyTorch needed)."""

import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="PokéDex Latent Space", page_icon="🔴", layout="wide")

TYPE_COLORS = {
    "fire": "#FF4136", "water": "#0074D9", "grass": "#2ECC40",
    "electric": "#FFDC00", "psychic": "#F012BE", "ice": "#7FDBFF",
    "dragon": "#7B68EE", "dark": "#333333", "fairy": "#FFB3DE",
    "normal": "#AAAAAA", "fighting": "#C0392B", "flying": "#89CFF0",
    "poison": "#9B59B6", "ground": "#D4A843", "rock": "#9B8651",
    "bug": "#8BC34A", "ghost": "#5C6BC0", "steel": "#78909C",
    "None": "#DDDDDD",
}

@st.cache_data
def load_data():
    df = pd.read_csv("export_data.csv")
    with open("manifest.json") as f:
        manifest = json.load(f)
    return df, manifest

df, manifest = load_data()

st.title("🔴 PokéDex Latent Space Explorer")

color_by = st.selectbox("Color by", ["type1", "cluster", "generation", "is_legendary"])

fig = go.Figure()
categories = df[color_by].unique()

for cat in categories:
    mask = df[color_by] == cat
    subset = df[mask]
    color = TYPE_COLORS.get(cat, "#AAAAAA")

    customdata = []
    for _, row in subset.iterrows():
        dex = str(int(row["pokedex_number"]))
        sprite = manifest.get(dex, {}).get("tooltip", "")
        customdata.append([row["name"], sprite, row["type1"], int(row["hp"]), int(row["attack"])])

    fig.add_trace(go.Scattergl(
        x=subset["emb_x"].values,
        y=subset["emb_y"].values,
        mode="markers",
        marker=dict(size=7, color=color, opacity=0.85, line=dict(width=0.5, color="white")),
        customdata=customdata,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "<img src=\'%{customdata[1]}\' width=\'72\'/><br>"
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
    height=600,
    margin=dict(l=40, r=40, t=30, b=40),
)

st.plotly_chart(fig, use_container_width=True)

# KNN panel
st.subheader("Click a Pokémon name to see similar ones")
selected_name = st.selectbox("Select Pokémon", df["name"].tolist())
if selected_name:
    idx = df[df["name"] == selected_name].index[0]
    row = df.iloc[idx]
    dex = str(int(row["pokedex_number"]))
    hero = manifest.get(dex, {}).get("card", "")
    st.image(hero, width=200)
    st.write(f"**{row[\'name\']}** — {row[\'type1\']}" + (f" / {row[\'type2\']}" if row.get("type2") != "None" else ""))
'''


def export_stlite_zip(
    df_export: pd.DataFrame,
    manifest: dict,
    output_path: str = "pokemon_latent_export.zip",
) -> str:
    """
    Write a ZIP containing:
    - export_data.csv (embeddings, cluster labels, sprite URLs, key stats)
    - index.html (stlite bootstrap)
    - app_export.py (pure Python/Plotly visualization)
    - manifest.json (sprite URL dict)
    Returns the path to the ZIP.
    """
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # CSV data
        csv_data = df_export.to_csv(index=False)
        zf.writestr("export_data.csv", csv_data)

        # Manifest
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        # HTML bootstrap
        zf.writestr("index.html", STLITE_INDEX_HTML)

        # Python app
        zf.writestr("app_export.py", APP_EXPORT_PY)

    return os.path.abspath(output_path)
