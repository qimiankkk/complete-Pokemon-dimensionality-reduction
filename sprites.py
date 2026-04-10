"""
Sprite URL resolution and manifest generation for the Pokémon dashboard.

All sprite URLs are deterministic from the Pokédex number. The manifest is
generated once and cached as sprite_manifest.json.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

SPRITE_BASE = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon"


def get_sprite_urls(dex_id: int) -> dict:
    """Return all sprite URL tiers for a given Pokédex number."""
    return {
        "tooltip": f"{SPRITE_BASE}/{dex_id}.png",
        "card": f"{SPRITE_BASE}/other/official-artwork/{dex_id}.png",
        "hero": f"{SPRITE_BASE}/other/home/{dex_id}.png",
        "shiny": f"{SPRITE_BASE}/other/official-artwork/shiny/{dex_id}.png",
    }


def _validate_url(url: str, timeout: float = 5.0) -> bool:
    """Check if a URL is reachable via HEAD request."""
    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def _validate_pokemon_sprites(dex_id: int) -> tuple[int, dict]:
    """Validate sprite URLs for a single Pokémon, falling back as needed."""
    urls = get_sprite_urls(dex_id)
    validated = {"tooltip": urls["tooltip"]}  # Tooltip sprites always exist

    # Validate higher-tier sprites; fall back to tooltip if missing
    for tier in ["card", "hero", "shiny"]:
        if _validate_url(urls[tier]):
            validated[tier] = urls[tier]
        else:
            validated[tier] = urls["tooltip"]

    return dex_id, validated


def build_sprite_manifest(
    df: pd.DataFrame,
    manifest_path: str = "sprite_manifest.json",
    max_workers: int = 20,
) -> dict:
    """
    Validate official-artwork URLs via HEAD requests (parallel).
    Falls back to tooltip sprite if artwork is missing.
    Writes result to manifest_path and returns the dict.
    """
    dex_ids = sorted(df["pokedex_number"].unique().tolist())
    manifest = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_validate_pokemon_sprites, dex_id): dex_id
            for dex_id in dex_ids
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Validating sprites"
        ):
            dex_id, validated = future.result()
            manifest[str(dex_id)] = validated

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def load_manifest(manifest_path: str = "sprite_manifest.json") -> dict | None:
    """Load an existing sprite manifest, or return None if it doesn't exist."""
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    df = pd.read_csv("pokemon.csv")
    manifest = build_sprite_manifest(df)
    print(f"Manifest generated with {len(manifest)} entries")

    # Verify a few random entries
    import random
    sample_ids = random.sample(list(manifest.keys()), min(5, len(manifest)))
    for dex_id in sample_ids:
        urls = manifest[dex_id]
        print(f"  #{dex_id}: tooltip={urls['tooltip'][:60]}...")
        print(f"          card={urls['card'][:60]}...")
