# PokeDex Latent Space Dashboard

An interactive Streamlit dashboard that applies nonlinear dimensionality reduction to the complete Pokemon dataset (801 Pokemon, 41 features), compressing base stats, type matchups, physical attributes, and breeding metadata into a 2D latent space via a configurable PyTorch Lightning autoencoder. The latent space is clustered with K-Means or HDBSCAN, rendered as a WebGL scatter plot with sprite tooltips, and enriched with official Pokemon artwork from the PokeAPI sprite CDN.

## Table of Contents

- [Quick Start](#quick-start)
- [User Journeys](#user-journeys)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Module Reference](#module-reference)
  - [data.py -- Data Layer](#datapy----data-layer)
  - [sprites.py -- Sprite Layer](#spritespy----sprite-layer)
  - [model.py -- Model Layer](#modelpy----model-layer)
  - [cluster.py -- Clustering Layer](#clusterpy----clustering-layer)
  - [viz.py -- Visualization Layer](#vizpy----visualization-layer)
  - [app.py -- Application Layer](#apppy----application-layer)
  - [exporter.py -- Export Layer](#exporterpy----export-layer)
- [Dashboard Layout](#dashboard-layout)
  - [Sidebar Controls](#sidebar-controls)
  - [Tab 1: Latent Space Explorer](#tab-1-latent-space-explorer)
  - [Tab 2: Pokemon Deep Dive](#tab-2-pokemon-deep-dive)
  - [Tab 3: EDA Explorer](#tab-3-eda-explorer)
- [Data Pipeline](#data-pipeline)
- [Model Architecture](#model-architecture)
- [Clustering Pipeline](#clustering-pipeline)
- [Sprite System](#sprite-system)
- [Dataset Notes](#dataset-notes)

---

## Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd complete-Pokemon-dimensionality-reduction

# Create virtual environment with uv
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Place pokemon.csv in the project root (download from Kaggle)
# https://www.kaggle.com/datasets/rounakbanik/pokemon/data

# Run the dashboard
streamlit run app.py
```

**Requirements:** Python 3.11+, `uv` (recommended) or `pip`.

---

## User Journeys

1. **Configure and train** -- select feature groups, model hyperparameters, and optimizer in the sidebar, then click "Train". The autoencoder trains with a live progress bar and early stopping.
2. **Explore the latent space** -- WebGL scatter plot renders all 801 Pokemon colored by type, cluster, generation, or legendary status. Convex hull overlays show cluster boundaries. Hovering shows the Pokemon's pixel sprite and key stats.
3. **Deep dive** -- clicking any point opens a panel with the Pokemon's official artwork, a hexagonal stat radar, a type-advantage heatmap, and a 10-neighbor KNN gallery strip.
4. **Raw EDA** -- a separate tab exposes histograms, 2D scatter plots, correlation heatmaps, type distributions, generation breakdowns, and legendary vs. regular violin comparisons.
5. **Export** -- one-click stlite WebAssembly ZIP for offline deployment (drag into Netlify for instant free hosting).

---

## Architecture

```
                  pokemon.csv
                      |
                  [ data.py ]          sklearn ColumnTransformer
                      |                (clean -> log1p -> scale -> OHE)
                      v
          float32 numpy array (N, D)
                      |
              +-------+-------+
              |               |
        [ model.py ]    [ model.py ]
        Autoencoder      PCA-only
        (PyTorch LT)     (sklearn)
              |               |
              v               v
         embeddings (N, 2)
                      |
                [ cluster.py ]         K-Means | HDBSCAN
                      |
              +-------+-------+--------+
              |       |       |        |
           labels   hulls  centroids  KNN
              |       |       |        |
              v       v       v        v
                  [ viz.py ]           Plotly figure builders
                      |
                  [ app.py ]           Streamlit 3-tab dashboard
                      |
                [ exporter.py ]        stlite WebAssembly ZIP
```

**Design principles:**
- **Flat module layout** -- all 7 Python files are sibling modules in the project root. No subdirectories, no package init files. Imports stay simple.
- **Strict layer separation** -- each module has a single responsibility and exposes a public API. `data.py` never imports `model.py`; `viz.py` never imports `cluster.py` directly.
- **Session state as the backbone** -- `app.py` persists all computed artifacts (model, embeddings, labels, hulls, manifest) in `st.session_state` so Streamlit reruns don't recompute expensive operations.

---

## Repository Structure

```
complete-Pokemon-dimensionality-reduction/
|-- app.py                  # Streamlit entry point, tab layout, sidebar, session state
|-- data.py                 # CSV loading, cleaning, sklearn preprocessing pipeline
|-- model.py                # PyTorch Lightning Autoencoder + PCA fallback
|-- cluster.py              # K-Means, HDBSCAN, convex hulls, KNN, cluster naming
|-- viz.py                  # Plotly figure builders (scatter, radar, heatmap, bars, violin)
|-- sprites.py              # PokeAPI sprite URL resolution + manifest validation
|-- exporter.py             # stlite WebAssembly ZIP packager
|-- pokemon.csv             # Source dataset (801 Pokemon, 41 features)
|-- requirements.txt        # Pinned Python dependencies
|-- sprite_manifest.json    # Auto-generated on first run (cached sprite URLs)
|-- .gitignore
|-- LICENSE
`-- README.md
```

---

## Module Reference

### `data.py` -- Data Layer

Loads `pokemon.csv`, applies cleaning rules, and builds a configurable sklearn preprocessing pipeline.

**Constants:**

| Name | Description |
|------|-------------|
| `TYPES` | List of 18 Pokemon type strings matching the `against_*` column suffixes |
| `FEATURE_GROUPS` | Dict mapping 5 group names to their column lists -- used to populate the sidebar multiselect |
| `LOG_COLS` | Set of columns that receive `log1p` transformation before scaling: `weight_kg`, `height_m`, `base_egg_steps`, `experience_growth` |

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `_clean_dataframe` | `(df: DataFrame) -> DataFrame` | Strips non-numeric chars from `capture_rate`, fills `percentage_male` nulls with -1.0, fills `type2` nulls with `"None"`, fills `height_m`/`weight_kg` nulls with median, drops rows where all 6 base stats are zero |
| `build_pipeline` | `(selected_groups: list[str], df: DataFrame) -> ColumnTransformer` | Constructs a sklearn `ColumnTransformer` with three branches: `StandardScaler` for regular numeric columns, `log1p` + `StandardScaler` for log-transformed columns, and `OneHotEncoder` for categorical columns |
| `load_and_preprocess` | `(csv_path: str, selected_groups: list[str]) -> tuple[ndarray, DataFrame, ColumnTransformer]` | Full pipeline: load CSV, clean, fit/transform. Returns the processed float32 array `(N, D)`, the original cleaned dataframe aligned row-for-row, and the fitted pipeline |

**Pipeline structure:**

```
ColumnTransformer
|-- ("num",     StandardScaler,                    [hp, attack, defense, ...])
|-- ("log_num", log1p -> StandardScaler,           [weight_kg, height_m, ...])
`-- ("cat",     OneHotEncoder(sparse_output=False), [type1, type2, generation, is_legendary])
```

---

### `sprites.py` -- Sprite Layer

Resolves PokeAPI sprite URLs from Pokedex numbers. Zero runtime API calls during normal operation -- all URLs are deterministic and validated once at startup.

**Constants:**

| Name | Value |
|------|-------|
| `SPRITE_BASE` | `https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon` |

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `get_sprite_urls` | `(dex_id: int) -> dict` | Returns 4 URL tiers: `tooltip` (96px pixel art), `card` (475px official artwork), `hero` (680px Home 3D render), `shiny` (shiny official artwork) |
| `_validate_url` | `(url: str, timeout: float) -> bool` | HEAD request to check reachability |
| `_validate_pokemon_sprites` | `(dex_id: int) -> tuple[int, dict]` | Validates all tiers for one Pokemon, falling back to tooltip sprite for any missing artwork |
| `build_sprite_manifest` | `(df: DataFrame, manifest_path: str, max_workers: int) -> dict` | Validates all 801 Pokemon sprites in parallel via `ThreadPoolExecutor` (20 workers), writes `sprite_manifest.json`, returns the manifest dict |
| `load_manifest` | `(manifest_path: str) -> dict or None` | Loads cached manifest from disk if it exists |

**Sprite tiers and where they appear:**

| Tier | Resolution | Used In |
|------|-----------|---------|
| `tooltip` | 96px pixel art | Plotly scatter hover, cluster centroid labels |
| `card` | ~475px official artwork | KNN gallery cards, EDA thumbnails |
| `hero` | ~680px Home 3D render | Deep dive hero image |
| `shiny` | ~475px shiny artwork | Deep dive shiny toggle |

---

### `model.py` -- Model Layer

PyTorch Lightning autoencoder for compressing the preprocessed feature matrix into a 2D latent space.

**Classes:**

**`Autoencoder(pl.LightningModule)`**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | `int` | -- | Number of input features (set automatically from data) |
| `hidden_layers` | `list[int]` | `[256, 128, 64]` | Hidden layer sizes for both encoder and decoder |
| `latent_dim` | `int` | `2` | Bottleneck dimension (fixed at 2 for visualization) |
| `activation` | `str` | `"relu"` | Activation function: `"relu"`, `"tanh"`, or `"gelu"` |
| `denoising_factor` | `float` | `0.0` | Gaussian noise stddev added to input during training (denoising autoencoder) |
| `optimizer_name` | `str` | `"adam"` | Optimizer: `"adam"` or `"sgd"` |
| `learning_rate` | `float` | `1e-3` | Learning rate |

Key design choice: the bottleneck layer has **no activation function**, leaving the latent space unbounded. This prevents collapse when compressing high-dimensional data into 2D.

Encoder architecture: `Linear -> Act -> Linear -> Act -> ... -> Linear (no act)`.
Decoder architecture: mirrors the encoder in reverse.

| Method | Description |
|--------|-------------|
| `encode(x)` | Forward pass through encoder only, returns 2D latent vectors |
| `decode(z)` | Forward pass through decoder only, returns reconstructed features |
| `forward(x)` | Full encode-decode pass |
| `training_step(batch, batch_idx)` | Adds Gaussian noise if `denoising_factor > 0`, computes MSE reconstruction loss |
| `configure_optimizers()` | Returns Adam or SGD based on config |

**`LossHistoryCallback(pl.Callback)`**

Records per-epoch training loss into a list for the sidebar live loss chart. Optionally calls a `progress_callback(epoch, max_epochs)` for the progress bar.

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `train_autoencoder` | `(X: ndarray, config: dict, loss_history: list, progress_callback) -> tuple[Autoencoder, ndarray]` | Creates model, DataLoader, Trainer with EarlyStopping. Returns trained model and embeddings `(N, 2)` |
| `apply_pca_orthogonalization` | `(embeddings: ndarray) -> ndarray` | Post-hoc PCA on the 2D latent space to force axis orthogonality |
| `run_pca_only` | `(X: ndarray) -> ndarray` | Bypass mode: sklearn PCA directly on the full feature matrix to 2D |

**Config dict keys for `train_autoencoder`:**

```python
{
    "hidden_layers": [256, 128, 64],
    "latent_dim": 2,
    "activation": "relu",
    "denoising_factor": 0.0,
    "optimizer_name": "adam",
    "learning_rate": 1e-3,
    "batch_size": 64,
    "max_epochs": 200,
    "patience": 10,
}
```

---

### `cluster.py` -- Clustering Layer

Two clustering algorithms, convex hull computation, KNN search, and heuristic cluster naming.

**Functions:**

| Function | Signature | Returns | Description |
|----------|-----------|---------|-------------|
| `run_kmeans` | `(embeddings, k) -> (labels, silhouette)` | `(ndarray, float)` | K-Means with `n_init=10` and silhouette score |
| `run_hdbscan` | `(embeddings, min_cluster_size, min_samples) -> (labels, n_clusters)` | `(ndarray, int)` | HDBSCAN clustering. Label `-1` = noise/outlier |
| `get_cluster_centroids` | `(embeddings, labels, df) -> DataFrame` | `DataFrame` | For each cluster, finds the Pokemon nearest to the geometric centroid. Returns `cluster_id`, `centroid_x`, `centroid_y`, `representative_name`, `dex_id` |
| `get_cluster_hulls` | `(embeddings, labels) -> dict[int, ndarray]` | `dict` | Computes `scipy.spatial.ConvexHull` for each cluster (skips clusters with < 3 points and noise label -1). Returns closed polygon vertices for Plotly |
| `find_knn` | `(embeddings, query_idx, k) -> list[int]` | `list[int]` | K nearest neighbors via sklearn `NearestNeighbors` (Euclidean). Excludes the query point itself |
| `name_cluster` | `(centroid_stats, legendary_frac) -> str` | `str` | Generates evocative cluster names from stat profiles |

**Cluster naming heuristics:**

| Condition | Name |
|-----------|------|
| > 50% legendaries | "Legendary Tier" |
| Total stats > 550 | "Pseudo-Legendary" |
| High atk + speed, low def | "Glass Cannon" |
| High def + HP, low atk | "Bulky Wall" |
| Speed dominates | "Speed Demon" |
| High HP + def, low atk | "Tank" |
| High atk + def | "Bruiser" |
| Total < 350 | "Underdog" |
| Low stat std dev | "All-Rounder" |
| Default | "Versatile" |

---

### `viz.py` -- Visualization Layer

All Plotly figure builders. Every function returns a `go.Figure` ready for `st.plotly_chart()`.

**Constants:**

| Name | Description |
|------|-------------|
| `TYPE_COLORS` | Dict mapping all 18 Pokemon types (+ `"None"`) to canonical hex colors |
| `CLUSTER_COLORS` | Combined Plotly + D3 qualitative palette (supports up to 20 clusters) |

**Figure builders:**

| Function | Purpose | Key Details |
|----------|---------|-------------|
| `build_latent_scatter(df_display, color_by, spotlight, manifest, show_hulls, hull_data)` | Main WebGL scatter plot | One `go.Scattergl` trace per category for legend/spotlight. Hull overlays as `go.Scatter(fill="toself")`. Hover template includes inline `<img>` sprite. Spotlight dims non-selected to 10% opacity |
| `build_stat_radar(pokemon_row, cluster_mean)` | Hexagonal stat radar | Two `go.Scatterpolar` traces: the Pokemon (filled, semi-transparent) and cluster average (dashed outline) |
| `build_type_advantage_heatmap(pokemon_row)` | 18-cell horizontal heatmap | Custom diverging color scale: 0x=blue, 0.5x=light blue, 1x=white, 2x=orange, 4x=red. Cell labels show multiplier value |
| `build_knn_gallery_html(neighbor_rows, manifest)` | HTML flex row of neighbor cards | Returns raw HTML string rendered via `st.components.v1.html()`. Each card: 96px sprite + name + type color badge. `onerror` fallback to tooltip sprite |
| `build_histogram(df, col, color_by)` | 1D distribution | `px.histogram` with optional color grouping by type or generation |
| `build_2d_scatter(df, x, y, color, trendline)` | 2D correlation scatter | `px.scatter` with WebGL rendering and optional OLS trendline |
| `build_correlation_matrix(df, cols)` | Pearson correlation heatmap | `RdBu_r` diverging scale, annotated cell values |
| `build_type_distribution(df, type_col)` | Horizontal bar chart of type counts | Bars colored with canonical type colors |
| `build_generation_breakdown(df, stacked)` | Generation x Type stacked/grouped bars | `pd.crosstab` then one `go.Bar` trace per type |
| `build_legendary_violin(df)` | Legendary vs regular violin plots | Side-by-side violins for all 6 base stats with box overlays |

---

### `app.py` -- Application Layer

Streamlit entry point. Manages session state, sidebar controls, three tabs, and click event handling.

**Session state keys:**

| Key | Type | Description |
|-----|------|-------------|
| `model` | `Autoencoder or None` | Trained PyTorch model |
| `embeddings` | `ndarray (N, 2) or None` | 2D latent embeddings |
| `df_display` | `DataFrame or None` | Original data augmented with `emb_x`, `emb_y`, `cluster` |
| `df_raw` | `DataFrame or None` | Cleaned raw data for EDA tab |
| `cluster_labels` | `ndarray (N,) or None` | Cluster assignments |
| `hull_data` | `dict or None` | Convex hull vertices per cluster |
| `manifest` | `dict or None` | Sprite URL manifest |
| `selected_point` | `int or None` | Index of clicked Pokemon |
| `loss_history` | `list[float]` | Per-epoch training loss for live chart |
| `pipeline` | `ColumnTransformer or None` | Fitted preprocessing pipeline |
| `trained` | `bool` | Whether a model has been trained |
| `clustered` | `bool` | Whether clustering has been run |

**Key functions:**

| Function | Description |
|----------|-------------|
| `load_raw_data(csv_path)` | `@st.cache_data` -- loads and cleans CSV once |
| `ensure_manifest(df)` | Loads manifest from disk or builds it with spinner on first run |
| `render_sidebar(df_raw)` | All sidebar controls: feature groups, model config, training, clustering, export |
| `render_tab_latent()` | Tab 1: scatter plot with color-by/spotlight/hull controls, click event handling, cluster representatives |
| `render_tab_deep_dive()` | Tab 2: hero image, stat radar, type heatmap, KNN gallery, cluster summary |
| `render_tab_eda()` | Tab 3: six expandable sections for exploratory analysis |
| `main()` | Top-level orchestrator: load data, ensure manifest, render sidebar, render tabs |

**Click event handling:**

Click events are captured via `streamlit-plotly-events`. Since Plotly traces are added per category (with hull traces first), the click handler maps `(curveNumber, pointIndex)` back to a global dataframe index by:
1. Subtracting the hull trace offset from `curveNumber`
2. Looking up which category the adjusted curve index corresponds to
3. Filtering the dataframe by that category and indexing into the subset

---

### `exporter.py` -- Export Layer

Packages the current dashboard state into a self-contained stlite WebAssembly app.

**Function:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `export_stlite_zip` | `(df_export: DataFrame, manifest: dict, output_path: str) -> str` | Writes a ZIP with 4 files and returns the absolute path |

**ZIP contents:**

| File | Purpose |
|------|---------|
| `export_data.csv` | Embeddings, cluster labels, key stats, type info |
| `manifest.json` | Sprite URL dict for all 801 Pokemon |
| `index.html` | stlite bootstrap HTML (loads Pyodide + Streamlit WASM runtime) |
| `app_export.py` | Standalone Python visualization app -- uses only Plotly and Pandas, no PyTorch |

The exported app can be deployed by dragging the unzipped folder into Netlify or any static hosting provider.

---

## Dashboard Layout

### Sidebar Controls

The sidebar is organized into five sections:

| Section | Controls |
|---------|----------|
| **Features** | Multiselect of 5 feature groups: Base Stats, Physical, Type Matchups, Breeding, Categorical |
| **Model** | Mode toggle (Autoencoder / PCA only), hidden layer config, activation (relu/tanh/gelu), optimizer (adam/sgd), denoising factor, learning rate, batch size, PCA orthogonalize checkbox |
| **Training** | Max epochs slider (50-500), early stopping patience (5-50), Train button, live loss chart |
| **Clustering** | Method toggle (K-Means / HDBSCAN), K slider (2-20) or min cluster size/min samples, Run Clustering button |
| **Export** | Export WebAssembly button, download ZIP |

### Tab 1: Latent Space Explorer

- **Color by** dropdown: `type1`, `cluster`, `generation`, `is_legendary`
- **Spotlight** dropdown: highlights one category at full opacity, dims all others to 10%
- **Show Hulls** checkbox: toggles convex hull overlays per cluster
- **WebGL scatter plot**: ~801 points with inline sprite tooltips on hover
- **Cluster representatives**: row of sprite thumbnails showing the Pokemon closest to each cluster centroid

### Tab 2: Pokemon Deep Dive

Activated by clicking a point on the Latent Space tab.

**Left column (40%):**
- Hero image (680px Home 3D render) with shiny toggle
- Name, Pokedex number, type badges (colored), generation, legendary status
- Cluster ID, auto-generated cluster name, top 3 Pokemon in cluster by base stat total

**Right column (60%):**
- Hexagonal stat radar: Pokemon stats vs. cluster average (dashed overlay)
- Type advantage heatmap: 18-cell row showing damage multipliers with diverging color scale

**Bottom (full width):**
- KNN gallery: 10 nearest neighbors in latent space, rendered as an HTML flex row of sprite cards

### Tab 3: EDA Explorer

Six expandable sections:

| Section | Chart Type | Controls |
|---------|-----------|----------|
| 1D Distributions | Histogram | Feature selector, optional color-by (type/generation), mean/median/std metrics |
| 2D Correlations | Scatter | X/Y axis selectors, color selector, optional OLS trendline |
| Type Distribution | Horizontal bar | Toggle between type1 and type2 |
| Generation Breakdown | Stacked/grouped bar | Type composition per generation, stacked toggle |
| Correlation Matrix | Annotated heatmap | Multiselect features, RdBu diverging scale |
| Legendary Spotlight | Violin plots | Side-by-side legendary vs. regular for all 6 base stats |

---

## Data Pipeline

```
pokemon.csv (801 rows, 41 columns)
    |
    v
_clean_dataframe()
    |-- capture_rate: strip "(Magnemite)" strings, cast to float
    |-- percentage_male: fill NaN with -1.0 (genderless sentinel)
    |-- type2: fill NaN with "None"
    |-- height_m, weight_kg: fill NaN with column median
    |-- Drop rows where all 6 base stats are 0
    v
build_pipeline(selected_groups)
    |-- Regular numeric cols -> StandardScaler
    |-- Log cols (weight_kg, height_m, base_egg_steps, experience_growth) -> log1p -> StandardScaler
    |-- Categorical cols (type1, type2, generation, is_legendary) -> OneHotEncoder
    v
float32 ndarray (801, D)    # D varies by selected feature groups: 6 to 76
```

**Feature group dimensions:**

| Group | Columns | D |
|-------|---------|---|
| Base Stats | hp, attack, defense, sp_attack, sp_defense, speed | 6 |
| Physical | height_m, weight_kg | 2 |
| Type Matchups | against_bug, against_dark, ... (18 types) | 18 |
| Breeding | base_happiness, capture_rate, base_egg_steps, experience_growth | 4 |
| Categorical | type1, type2, generation, is_legendary | ~46 (after OHE) |
| **All groups** | | **~76** |

---

## Model Architecture

**Autoencoder (default: `[256, 128, 64]` hidden layers):**

```
Encoder:                              Decoder (mirror):
  Input (D) ----+                       Latent (2) ----+
  Linear(D, 256) |                      Linear(2, 64)  |
  ReLU           |                      ReLU           |
  Linear(256, 128)|                     Linear(64, 128)|
  ReLU           |                      ReLU           |
  Linear(128, 64)|                      Linear(128, 256)|
  ReLU           |                      ReLU           |
  Linear(64, 2)  | <-- NO activation    Linear(256, D) |
  Latent (2) ----+                      Output (D) ----+
```

**Training loop:**
- Loss: MSE reconstruction loss
- Optional Gaussian noise injection (denoising autoencoder)
- Early stopping on `train_loss` with configurable patience
- `LossHistoryCallback` feeds epoch losses to the sidebar live chart
- Accelerator: auto (uses MPS on Apple Silicon, CUDA on NVIDIA, CPU otherwise)

**PCA modes:**
- **PCA orthogonalize**: post-hoc PCA on the 2D latent space to decorrelate the two axes
- **PCA only**: bypasses PyTorch entirely, applies sklearn PCA directly to the preprocessed feature matrix

---

## Clustering Pipeline

**K-Means:**
- `n_init=10`, `random_state=42` for reproducibility
- Silhouette score reported in sidebar
- K slider capped at 20

**HDBSCAN:**
- Density-based, automatically determines cluster count
- Label `-1` = noise/outlier (rendered as gray, excluded from hulls and centroids)
- Configurable `min_cluster_size` (3-30) and `min_samples` (1-15)

**Post-clustering:**
- Convex hulls computed via `scipy.spatial.ConvexHull` (polygons closed for Plotly)
- Cluster centroids: geometric mean of points, nearest Pokemon used as representative
- Cluster names auto-generated from centroid stat profiles

---

## Sprite System

All sprite URLs are deterministic from the Pokedex number -- no runtime API calls needed.

**First-run behavior:**
1. App checks for `sprite_manifest.json` on disk
2. If missing, spawns 20 parallel threads to HEAD-validate ~3200 URLs (4 tiers x 801 Pokemon)
3. Falls back to 96px pixel art for any missing higher-tier artwork
4. Writes validated manifest to disk for subsequent runs

**Manifest structure:**
```json
{
  "25": {
    "tooltip": "https://raw.githubusercontent.com/.../25.png",
    "card": "https://raw.githubusercontent.com/.../official-artwork/25.png",
    "hero": "https://raw.githubusercontent.com/.../home/25.png",
    "shiny": "https://raw.githubusercontent.com/.../official-artwork/shiny/25.png"
  }
}
```

---

## Dataset Notes

**Source:** [The Complete Pokemon Dataset](https://www.kaggle.com/datasets/rounakbanik/pokemon/data) by Rounak Banik -- 801 Pokemon, 41 columns.

**Known quirks in the raw CSV:**

| Column | Issue | Handling |
|--------|-------|----------|
| `capture_rate` | Contains strings like `"30 (Magnemite)"` | Regex strip + `pd.to_numeric` |
| `percentage_male` | NaN for genderless Pokemon | Fill with -1.0 (distinct sentinel) |
| `type2` | NaN for single-type Pokemon (65 rows) | Fill with `"None"` before OHE |
| `height_m` / `weight_kg` | Occasional NaN | Fill with column median |
| `classfication` | Typo in column name (not `classification`) | Not used in feature pipeline |
| `base_egg_steps` | Plan references `baseeggsteps` | Actual column name is `base_egg_steps` |
| `against_fight` | Plan references `against_fighting` | Actual column suffix is `fight` |
