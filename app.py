"""
Streamlit entry point — PokéDex Latent Space Dashboard.

Three tabs:
  1. Latent Space Explorer — WebGL scatter, clustering, hulls
  2. Pokémon Deep Dive     — hero image, radar, type heatmap, KNN gallery
  3. EDA Explorer          — histograms, scatter, correlation, violin
"""

import json
import os

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_plotly_events import plotly_events

import data as data_mod
import sprites as sprites_mod
import model as model_mod
import cluster as cluster_mod
import viz as viz_mod
from exporter import export_stlite_zip

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PokéDex Latent Space",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ───────────────────────────────────────────────────
DEFAULTS = {
    "model": None,
    "embeddings": None,
    "df_display": None,
    "df_raw": None,
    "cluster_labels": None,
    "hull_data": None,
    "manifest": None,
    "selected_point": None,
    "loss_history": [],
    "pipeline": None,
    "trained": False,
    "clustered": False,
}
for key, val in DEFAULTS.items():
    st.session_state.setdefault(key, val)


# ── Helper: load data + sprites on first run ─────────────────────────────────
@st.cache_data
def load_raw_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Apply cleaning
    return data_mod._clean_dataframe(df)


def ensure_manifest(df: pd.DataFrame) -> dict:
    """Load or build the sprite manifest."""
    if st.session_state.manifest is not None:
        return st.session_state.manifest

    manifest = sprites_mod.load_manifest()
    if manifest is not None:
        st.session_state.manifest = manifest
        return manifest

    with st.spinner("Building sprite manifest (first run only — validating URLs)..."):
        manifest = sprites_mod.build_sprite_manifest(df)
        st.session_state.manifest = manifest
        return manifest


# ── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar(df_raw: pd.DataFrame):
    with st.sidebar:
        st.markdown("## 🔴 PokéDex Latent Space")

        # ── Features ──
        st.markdown("### Features")
        selected_groups = st.multiselect(
            "Feature groups",
            list(data_mod.FEATURE_GROUPS.keys()),
            default=list(data_mod.FEATURE_GROUPS.keys()),
            label_visibility="collapsed",
        )
        if not selected_groups:
            st.warning("Select at least one feature group.")
            return None

        # ── Model ──
        st.markdown("### Model")
        mode = st.radio("Mode", ["Autoencoder", "PCA only"], horizontal=True)

        hidden_str = st.text_input("Hidden layers", "256,128,64")
        hidden_layers = [int(x.strip()) for x in hidden_str.split(",") if x.strip()]

        col1, col2 = st.columns(2)
        with col1:
            activation = st.selectbox("Activation", ["relu", "tanh", "gelu"])
        with col2:
            optimizer_name = st.selectbox("Optimizer", ["adam", "sgd"])

        denoising = st.slider("Denoising factor", 0.0, 0.5, 0.0, 0.05)
        learning_rate = st.select_slider(
            "Learning rate",
            options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
            value=1e-3,
            format_func=lambda x: f"{x:.0e}",
        )
        batch_size = st.slider("Batch size", 16, 256, 64, 16)
        pca_ortho = st.checkbox("PCA orthogonalize latent", value=False)

        # ── Training ──
        st.markdown("### Training")
        max_epochs = st.slider("Max epochs", 50, 500, 200, 10)
        patience = st.slider("Early stopping patience", 5, 50, 10, 5)

        train_btn = st.button("🚀 Train Model", use_container_width=True)

        if st.session_state.loss_history:
            st.line_chart(
                pd.DataFrame({"loss": st.session_state.loss_history}),
                height=120,
            )

        # ── Clustering ──
        st.markdown("### Clustering")
        cluster_method = st.radio("Method", ["K-Means", "HDBSCAN"], horizontal=True)
        if cluster_method == "K-Means":
            k = st.slider("K (clusters)", 2, 20, 8)
        else:
            min_cluster_size = st.slider("Min cluster size", 3, 30, 5)
            min_samples = st.slider("Min samples", 1, 15, 3)

        cluster_btn = st.button("🔬 Run Clustering", use_container_width=True)

        # ── Export ──
        st.markdown("### Export")
        export_btn = st.button("📦 Export WebAssembly", use_container_width=True)

    # ── Handle training ──
    config = {
        "hidden_layers": hidden_layers,
        "activation": activation,
        "denoising_factor": denoising,
        "optimizer_name": optimizer_name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
    }

    if train_btn:
        X, df_clean, pipeline = data_mod.load_and_preprocess(
            "pokemon.csv", selected_groups
        )
        st.session_state.pipeline = pipeline
        st.session_state.df_raw = df_clean

        if mode == "PCA only":
            with st.sidebar:
                with st.spinner("Running PCA..."):
                    embeddings = model_mod.run_pca_only(X)
                    st.session_state.model = None
                    st.session_state.loss_history = []
        else:
            loss_history = []
            progress_bar = st.sidebar.progress(0, text="Training...")

            def progress_cb(epoch, total):
                progress_bar.progress(
                    min((epoch + 1) / total, 1.0),
                    text=f"Epoch {epoch+1}/{total}",
                )

            model, embeddings = model_mod.train_autoencoder(
                X, config, loss_history, progress_cb
            )
            st.session_state.model = model
            st.session_state.loss_history = loss_history
            progress_bar.empty()

        if pca_ortho and mode != "PCA only":
            embeddings = model_mod.apply_pca_orthogonalization(embeddings)

        st.session_state.embeddings = embeddings

        # Build df_display with embedding coords
        df_display = df_clean.copy()
        df_display["emb_x"] = embeddings[:, 0]
        df_display["emb_y"] = embeddings[:, 1]
        df_display["cluster"] = -1
        st.session_state.df_display = df_display
        st.session_state.trained = True
        st.session_state.clustered = False
        st.session_state.selected_point = None

        st.rerun()

    # ── Handle clustering ──
    if cluster_btn and st.session_state.embeddings is not None:
        emb = st.session_state.embeddings
        if cluster_method == "K-Means":
            labels, sil = cluster_mod.run_kmeans(emb, k)
            st.sidebar.metric("Silhouette Score", f"{sil:.3f}")
        else:
            labels, n_found = cluster_mod.run_hdbscan(
                emb, min_cluster_size, min_samples
            )
            st.sidebar.metric("Clusters Found", n_found)

        st.session_state.cluster_labels = labels
        st.session_state.df_display["cluster"] = labels
        st.session_state.hull_data = cluster_mod.get_cluster_hulls(emb, labels)
        st.session_state.clustered = True
        st.rerun()

    # ── Handle export ──
    if export_btn and st.session_state.df_display is not None:
        manifest = st.session_state.manifest or {}
        export_cols = [
            "name", "pokedex_number", "type1", "type2", "generation",
            "is_legendary", "hp", "attack", "defense", "sp_attack",
            "sp_defense", "speed", "emb_x", "emb_y", "cluster",
        ]
        df_export = st.session_state.df_display[
            [c for c in export_cols if c in st.session_state.df_display.columns]
        ]
        zip_path = export_stlite_zip(df_export, manifest)
        with open(zip_path, "rb") as f:
            st.sidebar.download_button(
                "⬇️ Download ZIP",
                f.read(),
                file_name="pokemon_latent_export.zip",
                mime="application/zip",
            )

    return selected_groups


# ── Tab 1: Latent Space Explorer ─────────────────────────────────────────────
def render_tab_latent():
    if not st.session_state.trained or st.session_state.df_display is None:
        st.info("👈 Configure features and click **Train Model** in the sidebar to begin.")
        return

    df_display = st.session_state.df_display
    manifest = st.session_state.manifest or {}

    # Controls
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        color_options = ["type1", "cluster", "generation", "is_legendary"]
        color_by = st.selectbox("Color by", color_options, key="color_by_latent")
    with col2:
        if color_by in df_display.columns:
            unique_vals = ["None"] + sorted(
                [str(v) for v in df_display[color_by].unique()]
            )
            spotlight = st.selectbox("Spotlight", unique_vals, key="spotlight_latent")
            if spotlight == "None":
                spotlight = None
        else:
            spotlight = None
    with col3:
        show_hulls = st.checkbox("Show Hulls", value=True, key="show_hulls")

    # Build the scatter
    fig = viz_mod.build_latent_scatter(
        df_display=df_display,
        color_by=color_by,
        spotlight=spotlight,
        manifest=manifest,
        show_hulls=show_hulls,
        hull_data=st.session_state.hull_data,
    )

    # Render with click events
    clicked = plotly_events(fig, click_event=True, key="latent_scatter")
    if clicked:
        point_idx = clicked[0].get("pointIndex")
        curve_idx = clicked[0].get("curveNumber", 0)

        # Map curve + point index back to df index
        # Traces are added per category, so we need to track which trace maps to which rows
        if color_by in df_display.columns:
            categories = df_display[color_by].unique()
            if color_by == "type1":
                categories = df_display["type1"].unique()
            elif color_by == "cluster":
                categories = sorted(df_display["cluster"].unique())
            elif color_by == "generation":
                categories = sorted(df_display["generation"].unique())
            elif color_by == "is_legendary":
                categories = [0, 1]

            # Account for hull traces offset
            hull_offset = len(st.session_state.hull_data) if (
                show_hulls and st.session_state.hull_data
            ) else 0
            data_curve = curve_idx - hull_offset

            if 0 <= data_curve < len(list(categories)):
                cat = list(categories)[data_curve]
                mask = df_display[color_by] == cat
                subset_indices = df_display.index[mask]
                if point_idx is not None and point_idx < len(subset_indices):
                    global_idx = subset_indices[point_idx]
                    st.session_state.selected_point = global_idx

    # Show cluster stats
    if st.session_state.clustered:
        centroids = cluster_mod.get_cluster_centroids(
            st.session_state.embeddings,
            st.session_state.cluster_labels,
            st.session_state.df_display,
        )
        st.markdown("**Cluster Representatives:**")
        cols = st.columns(min(len(centroids), 6))
        for i, (_, row) in enumerate(centroids.iterrows()):
            with cols[i % len(cols)]:
                dex = str(int(row["dex_id"]))
                sprite = manifest.get(dex, {}).get("tooltip", "")
                if sprite:
                    st.image(sprite, width=48)
                st.caption(f"C{int(row['cluster_id'])}: {row['representative_name']}")


# ── Tab 2: Pokémon Deep Dive ─────────────────────────────────────────────────
def render_tab_deep_dive():
    if st.session_state.selected_point is None:
        st.info("🖱️ Click any Pokémon on the **Latent Space** tab to begin your deep dive.")
        return

    df_display = st.session_state.df_display
    manifest = st.session_state.manifest or {}
    idx = st.session_state.selected_point
    pokemon = df_display.iloc[idx]
    dex = str(int(pokemon["pokedex_number"]))

    # ── Layout: two columns ──
    left, right = st.columns([2, 3])

    with left:
        # Hero image with shiny toggle
        shiny = st.toggle("✨ Shiny", value=False, key="shiny_toggle")
        tier = "shiny" if shiny else "hero"
        hero_url = manifest.get(dex, {}).get(tier, "")
        fallback_url = manifest.get(dex, {}).get("card", "")
        st.image(hero_url or fallback_url, width=280)

        st.markdown(f"### #{pokemon['pokedex_number']} {pokemon['name']}")

        # Type badges
        type1_color = viz_mod.TYPE_COLORS.get(pokemon["type1"], "#AAA")
        type2 = pokemon.get("type2", "None")
        badges = f'<span style="background:{type1_color};color:#fff;padding:4px 10px;border-radius:12px;font-size:14px;font-weight:bold">{pokemon["type1"].upper()}</span>'
        if type2 != "None":
            type2_color = viz_mod.TYPE_COLORS.get(type2, "#AAA")
            badges += f' <span style="background:{type2_color};color:#fff;padding:4px 10px;border-radius:12px;font-size:14px;font-weight:bold">{type2.upper()}</span>'
        st.markdown(badges, unsafe_allow_html=True)

        st.markdown(f"**Generation:** {int(pokemon['generation'])}")
        if pokemon.get("is_legendary"):
            st.markdown("⭐ **Legendary**")

        # Cluster info
        if st.session_state.clustered:
            cid = int(pokemon.get("cluster", -1))
            if cid != -1:
                cluster_mask = df_display["cluster"] == cid
                cluster_df = df_display[cluster_mask]
                stats = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
                centroid_stats = {s: cluster_df[s].mean() for s in stats}
                legendary_frac = cluster_df["is_legendary"].mean()
                cname = cluster_mod.name_cluster(centroid_stats, legendary_frac)

                st.markdown(f"**Cluster:** {cid} — *{cname}*")

                # Top 3 in cluster by base_total
                if "base_total" in cluster_df.columns:
                    top3 = cluster_df.nlargest(3, "base_total")[["name", "base_total"]]
                    st.markdown("**Top in cluster:**")
                    for _, r in top3.iterrows():
                        st.caption(f"  {r['name']} ({int(r['base_total'])})")

    with right:
        # Stat radar
        st.markdown("#### Stat Radar")
        cluster_mean = None
        if st.session_state.clustered:
            cid = int(pokemon.get("cluster", -1))
            if cid != -1:
                stats = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
                cluster_mask = df_display["cluster"] == cid
                cluster_mean = df_display[cluster_mask][stats].mean()
        radar_fig = viz_mod.build_stat_radar(pokemon, cluster_mean)
        st.plotly_chart(radar_fig, use_container_width=True)

        # Type advantage heatmap
        st.markdown("#### Type Effectiveness")
        heatmap_fig = viz_mod.build_type_advantage_heatmap(pokemon)
        st.plotly_chart(heatmap_fig, use_container_width=True)

    # KNN Gallery
    st.markdown("---")
    st.markdown("#### 🔗 10 Most Similar Pokémon")
    if st.session_state.embeddings is not None:
        neighbors = cluster_mod.find_knn(st.session_state.embeddings, idx, k=10)
        neighbor_rows = df_display.iloc[neighbors]
        gallery_html = viz_mod.build_knn_gallery_html(neighbor_rows, manifest)
        st.components.v1.html(gallery_html, height=160, scrolling=True)


# ── Tab 3: EDA Explorer ──────────────────────────────────────────────────────
def render_tab_eda():
    df_raw = st.session_state.df_raw
    if df_raw is None:
        # Load raw data anyway for EDA
        df_raw = load_raw_data("pokemon.csv")
        st.session_state.df_raw = df_raw

    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()

    # ── 1D Distributions ──
    with st.expander("📊 1D Distributions", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            hist_col = st.selectbox("Feature", numeric_cols, key="hist_col")
        with col2:
            hist_color = st.selectbox(
                "Color by", ["None", "type1", "generation"], key="hist_color"
            )
        color_arg = None if hist_color == "None" else hist_color
        fig = viz_mod.build_histogram(df_raw, hist_col, color_arg)
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean", f"{df_raw[hist_col].mean():.2f}")
        c2.metric("Median", f"{df_raw[hist_col].median():.2f}")
        c3.metric("Std", f"{df_raw[hist_col].std():.2f}")

    # ── 2D Correlations ──
    with st.expander("🔗 2D Correlations"):
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("X axis", numeric_cols, index=0, key="scatter_x")
        with col2:
            y_col = st.selectbox(
                "Y axis", numeric_cols,
                index=min(1, len(numeric_cols) - 1),
                key="scatter_y",
            )
        with col3:
            scatter_color = st.selectbox(
                "Color", ["type1", "generation", "is_legendary"], key="scatter_color"
            )
        trendline = st.checkbox("Show trendline", value=False, key="trendline")
        fig = viz_mod.build_2d_scatter(df_raw, x_col, y_col, scatter_color, trendline)
        st.plotly_chart(fig, use_container_width=True)

    # ── Type Distribution ──
    with st.expander("🏷️ Type Distribution"):
        type_col = st.radio("Type column", ["type1", "type2"], horizontal=True, key="type_dist")
        fig = viz_mod.build_type_distribution(df_raw, type_col)
        st.plotly_chart(fig, use_container_width=True)

    # ── Generation Breakdown ──
    with st.expander("📅 Generation Breakdown"):
        stacked = st.checkbox("Stacked", value=True, key="gen_stacked")
        fig = viz_mod.build_generation_breakdown(df_raw, stacked)
        st.plotly_chart(fig, use_container_width=True)

    # ── Correlation Matrix ──
    with st.expander("🔥 Correlation Matrix"):
        default_cols = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
        selected_corr = st.multiselect(
            "Features", numeric_cols,
            default=[c for c in default_cols if c in numeric_cols],
            key="corr_cols",
        )
        if len(selected_corr) >= 2:
            fig = viz_mod.build_correlation_matrix(df_raw, selected_corr)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Select at least 2 features.")

    # ── Legendary vs Non-Legendary ──
    with st.expander("⭐ Legendary Spotlight"):
        fig = viz_mod.build_legendary_violin(df_raw)
        st.plotly_chart(fig, use_container_width=True)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Load raw data for EDA (always available)
    df_raw = load_raw_data("pokemon.csv")
    if st.session_state.df_raw is None:
        st.session_state.df_raw = df_raw

    # Load/build sprite manifest
    ensure_manifest(df_raw)

    # Sidebar
    render_sidebar(df_raw)

    # Tabs
    tab_latent, tab_dive, tab_eda = st.tabs([
        "🌐 Latent Space",
        "🔍 Pokémon Deep Dive",
        "📊 EDA Explorer",
    ])

    with tab_latent:
        render_tab_latent()

    with tab_dive:
        render_tab_deep_dive()

    with tab_eda:
        render_tab_eda()


if __name__ == "__main__":
    main()
