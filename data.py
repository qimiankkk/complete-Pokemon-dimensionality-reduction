"""
Data loading and preprocessing pipeline for the Pokémon latent space dashboard.

Loads pokemon.csv, cleans it, and builds a configurable sklearn preprocessing
pipeline that returns tensor-ready numpy arrays.
"""

import re
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

TYPES = [
    "bug", "dark", "dragon", "electric", "fairy", "fight", "fire", "flying",
    "ghost", "grass", "ground", "ice", "normal", "poison", "psychic", "rock",
    "steel", "water",
]

FEATURE_GROUPS = {
    "Base Stats":    ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"],
    "Physical":      ["height_m", "weight_kg"],
    "Type Matchups": [f"against_{t}" for t in TYPES],
    "Breeding":      ["base_happiness", "capture_rate", "base_egg_steps", "experience_growth"],
    "Categorical":   ["type1", "type2", "generation", "is_legendary"],
}

# Columns that benefit from log1p transform before scaling
LOG_COLS = {"weight_kg", "height_m", "base_egg_steps", "experience_growth"}


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning rules to the raw dataframe."""
    df = df.copy()

    # capture_rate: strip non-numeric chars (e.g. "30 (Magnemite)") and cast
    df["capture_rate"] = df["capture_rate"].apply(
        lambda x: re.sub(r"\(.*\)", "", str(x)).strip()
    )
    df["capture_rate"] = pd.to_numeric(df["capture_rate"], errors="coerce")

    # percentage_male: fill genderless with -1
    df["percentage_male"] = df["percentage_male"].fillna(-1.0)

    # type2: fill NaN with "None" before OHE
    df["type2"] = df["type2"].fillna("None")

    # height_m / weight_kg: fill NaN with column median
    for col in ["height_m", "weight_kg"]:
        df[col] = df[col].fillna(df[col].median())

    # Drop rows where all 6 base stats are 0 (corrupt entries)
    base_stat_cols = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
    mask = df[base_stat_cols].sum(axis=1) > 0
    df = df[mask].reset_index(drop=True)

    return df


def build_pipeline(selected_groups: list[str], df: pd.DataFrame) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer for the selected feature groups.
    The df is needed to fit the OHE categories.
    """
    numeric_cols = []
    log_numeric_cols = []
    cat_cols = []

    for group in selected_groups:
        if group == "Categorical":
            cat_cols.extend(FEATURE_GROUPS["Categorical"])
        else:
            for col in FEATURE_GROUPS[group]:
                if col in LOG_COLS:
                    log_numeric_cols.append(col)
                else:
                    numeric_cols.append(col)

    transformers = []

    if numeric_cols:
        numeric_pipe = Pipeline([("scaler", StandardScaler())])
        transformers.append(("num", numeric_pipe, numeric_cols))

    if log_numeric_cols:
        log_pipe = Pipeline([
            ("log1p", FunctionTransformer(np.log1p, validate=True)),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("log_num", log_pipe, log_numeric_cols))

    if cat_cols:
        cat_pipe = Pipeline([
            ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers, remainder="drop")


def load_and_preprocess(
    csv_path: str,
    selected_groups: list[str],
) -> tuple[np.ndarray, pd.DataFrame, ColumnTransformer]:
    """
    Load and preprocess the Pokémon dataset.

    Returns:
        X_processed  — shape (N, D), float32 numpy array ready for autoencoder
        df_display   — original dataframe aligned row-for-row with X_processed
        pipeline     — fitted ColumnTransformer (needed for inverse_transform)
    """
    df = pd.read_csv(csv_path)
    df = _clean_dataframe(df)

    pipeline = build_pipeline(selected_groups, df)
    X_processed = pipeline.fit_transform(df).astype(np.float32)

    return X_processed, df, pipeline


if __name__ == "__main__":
    for groups in [
        ["Base Stats"],
        ["Base Stats", "Physical"],
        ["Base Stats", "Type Matchups", "Categorical"],
        list(FEATURE_GROUPS.keys()),
    ]:
        X, df, pipe = load_and_preprocess("pokemon.csv", groups)
        print(f"Groups: {groups}")
        print(f"  Shape: {X.shape}, dtype: {X.dtype}")
        print(f"  First row (5 features): {X[0, :5]}")
        print()
