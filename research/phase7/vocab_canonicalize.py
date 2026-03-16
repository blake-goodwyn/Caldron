"""
Phase 7 -- Vocabulary Canonicalization

Maps ingredient variants to canonical forms, removes noise tokens and
brand names. Improves all downstream ML models by reducing vocabulary
fragmentation.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Known noise tokens ───────────────────────────────────────────────────

SEED_BLOCKLIST = {
    "amounts", "containers", "pink", "new", "sweet", "regular",
    "skinless", "boneless", "frozen", "hot", "cold", "warm",
    "prepared", "cooked", "uncooked", "raw", "dry", "wet",
    "flat", "round", "long", "thin", "thick",
}

# ── Known brand prefixes ─────────────────────────────────────────────────

BRAND_PREFIXES = [
    "philadelphia", "parkay", "kraft", "pillsbury", "nabisco",
    "jell-o", "bisquick", "uncle ben", "old el paso", "frank",
    "hellmann", "heinz", "campbells", "lipton", "knorr",
    "mccormick", "lawry", "french", "pace", "tostitos",
    "realemon", "realime",
]

# ── Known compound dishes (not ingredients) ──────────────────────────────

SEED_COMPOUND_DISHES = {
    "hot buttered noodles", "broken spaghetti", "hot mashed potatoes",
    "macaroni and cheese", "peanut butter and jelly",
}


class CanonicalMap:
    """Maps ingredient variants to canonical forms.

    Three structures:
    - synonyms: variant -> canonical name
    - blocklist: tokens to remove entirely
    - compound_dishes: multi-word non-ingredients to remove
    """

    def __init__(
        self,
        synonyms: Optional[dict[str, str]] = None,
        blocklist: Optional[set[str]] = None,
        compound_dishes: Optional[set[str]] = None,
    ):
        self.synonyms = synonyms or {}
        self.blocklist = blocklist or set()
        self.compound_dishes = compound_dishes or set()

    def save(self, path: Path):
        data = {
            "synonyms": self.synonyms,
            "blocklist": sorted(self.blocklist),
            "compound_dishes": sorted(self.compound_dishes),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "CanonicalMap":
        with open(path) as f:
            data = json.load(f)
        return cls(
            synonyms=data.get("synonyms", {}),
            blocklist=set(data.get("blocklist", [])),
            compound_dishes=set(data.get("compound_dishes", [])),
        )


def canonicalize(ingredient: str, cmap: CanonicalMap) -> Optional[str]:
    """Map an ingredient to its canonical form.

    Returns None if the ingredient is in the blocklist or is a compound dish.
    Returns the canonical synonym if one exists, otherwise returns unchanged.
    """
    name = ingredient.strip().lower()

    if not name:
        return None

    # Check blocklist
    if name in cmap.blocklist:
        return None

    # Check compound dishes
    if name in cmap.compound_dishes:
        return None

    # Apply synonym mapping
    if name in cmap.synonyms:
        return cmap.synonyms[name]

    return name


# ── Automated canonical map builder ──────────────────────────────────────

def _strip_brand(name: str) -> Optional[str]:
    """Strip known brand prefixes from an ingredient name."""
    for brand in BRAND_PREFIXES:
        if name.startswith(brand + " "):
            stripped = name[len(brand):].strip()
            if len(stripped) > 1:
                return stripped
    return None


# Words that change the identity of an ingredient when added
# (i.e., "almond extract" is NOT a variant of "almond")
IDENTITY_CHANGING_SUFFIXES = {
    "extract", "sauce", "oil", "juice", "paste", "powder", "flour",
    "milk", "cream", "butter", "water", "stock", "broth", "vinegar",
    "syrup", "liqueur", "wine", "beer", "zest", "peel", "rind",
    "seeds", "seed", "leaves", "leaf", "bark", "root", "chips",
    "flakes", "meal", "starch", "puree", "jam", "jelly",
    "preserve", "filling", "frosting", "glaze", "seasoning",
    "dressing", "marinade", "cheese",
}

# Words that are just modifiers and don't change identity
KNOWN_MODIFIERS = {
    "unsalted", "salted", "sweet", "sour", "dry", "dried", "fresh",
    "frozen", "canned", "whole", "ground", "crushed", "chopped",
    "diced", "sliced", "minced", "grated", "shredded", "cubed",
    "peeled", "seeded", "pitted", "boneless", "skinless",
    "cooked", "uncooked", "raw", "roasted", "toasted", "smoked",
    "light", "lite", "low-fat", "nonfat", "fat-free",
    "extra", "virgin", "pure", "organic", "instant", "quick",
    "regular", "plain", "white", "dark", "red", "green", "yellow",
    "black", "long", "short", "baby", "jumbo", "large", "small",
    "medium", "thick", "thin",
}


def _is_modifier_variant(long_name: str, short_name: str) -> bool:
    """Check if the longer name is the shorter name with only modifier words added.

    Returns True for "unsalted butter" -> "butter" (modifier + base).
    Returns False for "almond extract" -> "almond" (extract changes identity).
    Returns False for "acorn" containing "corn" (not a word-boundary match).
    """
    import re

    # The short name must appear as complete words in the long name
    pattern = rf'\b{re.escape(short_name)}\b'
    if not re.search(pattern, long_name):
        return False

    # Get the extra words
    extra = re.sub(pattern, "", long_name).strip()
    if not extra:
        return False

    extra_words = set(extra.split())

    # If any extra word is an identity-changing suffix, reject
    if extra_words & IDENTITY_CHANGING_SUFFIXES:
        return False

    # All extra words should be known modifiers
    if not extra_words.issubset(KNOWN_MODIFIERS):
        return False

    return True


def _find_substring_synonyms(
    words: list[str],
    counts: dict[str, int],
    max_extra_chars: int = 15,
) -> dict[str, str]:
    """Find synonym pairs where one name is a modifier variant of another.

    Only merges when the short form appears as a complete word in the long
    form and constitutes a significant portion of the name.
    """
    synonyms = {}
    sorted_words = sorted(words, key=len)

    for i, short in enumerate(sorted_words):
        if len(short) < 3:
            continue
        for long in sorted_words[i + 1:]:
            if short == long:
                continue
            if short not in long:
                continue
            if (len(long) - len(short)) > max_extra_chars:
                continue

            # Only merge if it's a modifier relationship
            if not _is_modifier_variant(long, short):
                continue

            short_count = counts.get(short, 0)
            long_count = counts.get(long, 0)

            if short_count >= long_count:
                synonyms[long] = short
            elif long_count > short_count * 10:
                synonyms[short] = long
            else:
                synonyms[long] = short

    return synonyms


def build_canonical_map(
    vocab,
    overrides_path: Optional[Path] = None,
) -> CanonicalMap:
    """Build a canonical map from a fitted IngredientVocab.

    Args:
        vocab: Fitted IngredientVocab with word2idx and counter.
        overrides_path: Optional JSON file with manual overrides.

    Returns:
        CanonicalMap with synonyms, blocklist, and compound_dishes.
    """
    words = list(vocab.word2idx.keys())
    counts = dict(vocab.counter)

    # 1. Find substring-based synonyms
    synonyms = _find_substring_synonyms(words, counts)

    # 2. Strip brands
    for word in words:
        stripped = _strip_brand(word)
        if stripped and stripped in vocab.word2idx and word not in synonyms:
            synonyms[word] = stripped

    # 3. Build blocklist from seed + single-token heuristic
    blocklist = set()
    for word in words:
        if word in SEED_BLOCKLIST:
            blocklist.add(word)

    # 4. Compound dishes from seed
    compound_dishes = set()
    for word in words:
        if word in SEED_COMPOUND_DISHES:
            compound_dishes.add(word)

    # 5. Apply manual overrides
    if overrides_path and overrides_path.exists():
        with open(overrides_path) as f:
            overrides = json.load(f)
        synonyms.update(overrides.get("synonyms", {}))
        blocklist.update(overrides.get("blocklist", []))
        compound_dishes.update(overrides.get("compound_dishes", []))

    # Don't let canonical targets be in the blocklist
    canonical_targets = set(synonyms.values())
    blocklist -= canonical_targets

    cmap = CanonicalMap(
        synonyms=synonyms,
        blocklist=blocklist,
        compound_dishes=compound_dishes,
    )

    logger.info(
        f"Built canonical map: {len(synonyms)} synonyms, "
        f"{len(blocklist)} blocked, {len(compound_dishes)} compound dishes"
    )
    return cmap
