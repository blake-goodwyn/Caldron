"""
Phase 7 -- Fetch FlavorDB compound data.

Downloads pre-scraped FlavorDB CSVs from GitHub and converts to the
JSON format expected by data_pipeline.load_flavordb():
  {ingredient_name: [compound1, compound2, ...]}

Usage:
    python fetch_flavordb.py
"""

import csv
import io
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

# Pre-scraped FlavorDB data from wannasleepforlong/flavordb on GitHub
FLAVORDB_CSV_URL = "https://raw.githubusercontent.com/wannasleepforlong/flavordb/master/flavordb.csv"
MOLECULES_CSV_URL = "https://raw.githubusercontent.com/wannasleepforlong/flavordb/master/molecules.csv"


def fetch_url(url: str) -> str:
    """Fetch URL content as string."""
    logger.info(f"Fetching {url}")
    with urlopen(url) as response:
        return response.read().decode("utf-8")


def parse_pubchem_ids(id_string: str) -> list[str]:
    """Parse PubChem IDs from a string like '{123, 456, 789}'."""
    cleaned = id_string.strip().strip("{}").strip("[]")
    if not cleaned:
        return []
    return [s.strip().strip("'\"") for s in cleaned.split(",") if s.strip()]


def build_flavordb_json(output_path: Path = None) -> dict[str, list[str]]:
    """Download and build FlavorDB ingredient-to-compounds mapping.

    Returns:
        Dict mapping ingredient names to lists of compound common names.
    """
    output_path = output_path or (DATA_DIR / "flavordb.json")

    # 1. Fetch molecules CSV: PubChem ID -> common name + flavor profiles
    molecules_csv = fetch_url(MOLECULES_CSV_URL)
    pubchem_to_name = {}
    compound_profiles = {}  # compound_name -> set of flavor descriptors
    reader = csv.DictReader(io.StringIO(molecules_csv))
    for row in reader:
        pubchem_id = row.get("pubchem id", "").strip()
        common_name = row.get("common name", "").strip()
        flavor_profile_str = row.get("flavor profile", "").strip()
        if pubchem_id and common_name:
            name_lower = common_name.lower()
            pubchem_to_name[pubchem_id] = name_lower
            # Parse flavor profile from Python set notation: "{'sweet', 'umami'}"
            if flavor_profile_str:
                try:
                    import ast
                    profiles = ast.literal_eval(flavor_profile_str)
                    if isinstance(profiles, set):
                        compound_profiles[name_lower] = sorted(profiles)
                except (ValueError, SyntaxError):
                    pass

    logger.info(f"Loaded {len(pubchem_to_name)} compounds, "
                f"{len(compound_profiles)} with flavor profiles")

    # 2. Fetch flavordb CSV: ingredient -> set of PubChem IDs
    flavordb_csv = fetch_url(FLAVORDB_CSV_URL)
    ingredient_compounds = {}
    reader = csv.DictReader(io.StringIO(flavordb_csv))
    for row in reader:
        name = row.get("alias", "").strip().lower()
        if not name:
            continue

        # The molecules column contains PubChem IDs in {id1, id2, ...} format
        mol_str = row.get("molecules", "")
        pubchem_ids = parse_pubchem_ids(mol_str)

        compounds = []
        for pid in pubchem_ids:
            compound_name = pubchem_to_name.get(pid)
            if compound_name:
                compounds.append(compound_name)

        if compounds:
            ingredient_compounds[name] = sorted(set(compounds))

    logger.info(
        f"Built FlavorDB: {len(ingredient_compounds)} ingredients, "
        f"{sum(len(v) for v in ingredient_compounds.values())} total compound links"
    )

    # 3. Build ingredient flavor profiles (aggregate compound profiles)
    ingredient_profiles = {}
    for name, compounds in ingredient_compounds.items():
        descriptors = set()
        compound_details = {}
        for compound in compounds:
            if compound in compound_profiles:
                profiles = compound_profiles[compound]
                descriptors.update(profiles)
                compound_details[compound] = profiles
        if descriptors:
            ingredient_profiles[name] = {
                "descriptors": sorted(descriptors),
                "compounds_with_profiles": compound_details,
            }

    logger.info(f"Built flavor profiles for {len(ingredient_profiles)} ingredients")

    # 4. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(ingredient_compounds, f, indent=2, sort_keys=True)

    profiles_path = output_path.parent / "ingredient_flavor_profiles.json"
    with open(profiles_path, "w") as f:
        json.dump(ingredient_profiles, f, indent=2, sort_keys=True)

    logger.info(f"Saved compounds to {output_path}")
    logger.info(f"Saved flavor profiles to {profiles_path}")
    return ingredient_compounds


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = build_flavordb_json()
    print(f"\nFlavorDB: {len(result)} ingredients")

    # Show sample with flavor profiles
    profiles_path = DATA_DIR / "ingredient_flavor_profiles.json"
    if profiles_path.exists():
        with open(profiles_path) as f:
            profiles = json.load(f)
        samples = ["tomato", "basil", "garlic", "chocolate", "cinnamon"]
        for s in samples:
            if s in profiles:
                descs = profiles[s]["descriptors"][:8]
                print(f"  {s}: {', '.join(descs)} ({len(profiles[s]['descriptors'])} descriptors)")
