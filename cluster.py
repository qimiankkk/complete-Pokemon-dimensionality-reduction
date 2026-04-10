"""
Clustering layer: K-Means, HDBSCAN, convex hulls, and KNN search.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull
import hdbscan


def run_kmeans(embeddings: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """
    Run K-Means clustering on 2D embeddings.
    Returns (labels, silhouette_score).
    """
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(embeddings)
    sil = silhouette_score(embeddings, labels)
    return labels, sil


def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 3,
) -> tuple[np.ndarray, int]:
    """
    Run HDBSCAN on 2D embeddings.
    Returns (labels, n_clusters_found). Label -1 means noise/outlier.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, n_clusters


def get_cluster_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each cluster, find the Pokémon closest to the geometric centroid.
    Returns a dataframe with: cluster_id, centroid_x, centroid_y,
    representative_name, dex_id.
    """
    unique_labels = sorted(set(labels))
    rows = []
    for cid in unique_labels:
        if cid == -1:
            continue
        mask = labels == cid
        cluster_pts = embeddings[mask]
        centroid = cluster_pts.mean(axis=0)

        # Find closest point to centroid
        dists = np.linalg.norm(cluster_pts - centroid, axis=1)
        local_idx = np.argmin(dists)
        global_idx = np.where(mask)[0][local_idx]

        rows.append({
            "cluster_id": cid,
            "centroid_x": centroid[0],
            "centroid_y": centroid[1],
            "representative_name": df.iloc[global_idx]["name"],
            "dex_id": int(df.iloc[global_idx]["pokedex_number"]),
        })

    return pd.DataFrame(rows)


def get_cluster_hulls(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """
    Returns {cluster_id: hull_vertices_array} for Plotly polygon traces.
    Skips clusters with fewer than 3 points (can't form a hull).
    """
    hulls = {}
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        mask = labels == cid
        pts = embeddings[mask]
        if len(pts) < 3:
            continue
        try:
            hull = ConvexHull(pts)
            # Close the polygon by appending the first vertex
            vertices = pts[hull.vertices]
            vertices = np.vstack([vertices, vertices[0]])
            hulls[cid] = vertices
        except Exception:
            continue
    return hulls


def find_knn(
    embeddings: np.ndarray,
    query_idx: int,
    k: int = 10,
) -> list[int]:
    """Return indices of k nearest neighbors of the point at query_idx."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings[query_idx].reshape(1, -1))
    # Exclude the query point itself
    return indices[0, 1:].tolist()


def name_cluster(centroid_stats: dict, legendary_frac: float = 0.0) -> str:
    """
    Auto-generate evocative cluster names based on stat profile.

    centroid_stats: dict with keys hp, attack, defense, sp_attack, sp_defense, speed
    legendary_frac: fraction of legendaries in the cluster
    """
    hp = centroid_stats.get("hp", 0)
    attack = centroid_stats.get("attack", 0)
    defense = centroid_stats.get("defense", 0)
    sp_attack = centroid_stats.get("sp_attack", 0)
    sp_defense = centroid_stats.get("sp_defense", 0)
    speed = centroid_stats.get("speed", 0)

    total = hp + attack + defense + sp_attack + sp_defense + speed

    if legendary_frac > 0.5:
        return "Legendary Tier"
    if total > 550:
        return "Pseudo-Legendary"

    atk = max(attack, sp_attack)
    dfs = max(defense, sp_defense)

    if atk > dfs * 1.5 and speed > dfs:
        return "Glass Cannon"
    if dfs > atk * 1.3 and hp > 80:
        return "Bulky Wall"
    if speed > atk and speed > dfs:
        return "Speed Demon"
    if hp > 90 and dfs > 80 and atk < 70:
        return "Tank"
    if atk > 100 and dfs > 80:
        return "Bruiser"
    if total < 350:
        return "Underdog"

    # Check for balance
    stats = [hp, attack, defense, sp_attack, sp_defense, speed]
    std = np.std(stats)
    if std < 15:
        return "All-Rounder"

    return "Versatile"


if __name__ == "__main__":
    from data import load_and_preprocess
    from model import run_pca_only

    X, df, _ = load_and_preprocess("pokemon.csv", ["Base Stats"])
    embeddings = run_pca_only(X)
    print(f"Embeddings shape: {embeddings.shape}")

    # K-Means
    labels, sil = run_kmeans(embeddings, k=8)
    print(f"K-Means: {len(set(labels))} clusters, silhouette={sil:.3f}")

    # HDBSCAN
    labels_h, n = run_hdbscan(embeddings)
    print(f"HDBSCAN: {n} clusters found")

    # Centroids
    centroids = get_cluster_centroids(embeddings, labels, df)
    print(f"Centroids:\n{centroids[['cluster_id', 'representative_name']].to_string()}")

    # Hulls
    hulls = get_cluster_hulls(embeddings, labels)
    print(f"Hulls computed for {len(hulls)} clusters")

    # KNN
    neighbors = find_knn(embeddings, 0, k=5)
    print(f"5 nearest neighbors of {df.iloc[0]['name']}: {[df.iloc[i]['name'] for i in neighbors]}")
