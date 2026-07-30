"""
Compare the Senegal entries of the global thermal/renewable power plant
inventory (Supplementary-Data-1_INVENTORY.xlsx) against the
PyPSA-Earth-curated power plant database (resources/powerplants.csv).

Why this is non-trivial
------------------------
The two sources describe the same physical fleet very differently:

- ``data/validation/Supplementary-Data-1_INVENTORY.xlsx`` (sheet
  ``Main_Inventory``) lists individual *generating units* (e.g. each engine
  at "CAP DES BICHES C1/C2/C3/C4", every small village diesel set), with a
  ``Country`` column, free-text ``Plant`` names, capacity in ``MW`` and
  coordinates in ``Longitude``/``Latitude``.
- ``resources/powerplants.csv`` lists *sites* (often the sum of several
  units, e.g. one "Cap Des Biches" row), with capacity in ``Capacity`` and
  coordinates in ``lat``/``lon``.

Names rarely match exactly ("BEL-AIR" vs. "Bel Air", "TAIBA NDIAYE" vs.
"Taiba Diaye Wind Farm") and a handful of coordinates in both files look
mis-entered. So matching is done heuristically using a blend of:

1. Normalised name similarity (token overlap + character-level ratio).
2. Geographic proximity (haversine distance in km).

Every inventory row is matched independently to its best-scoring
powerplants.csv row (many inventory rows -> one powerplants.csv row is
expected and fine, since powerplants.csv is more aggregated). Matches below
the score threshold are reported as unmatched rather than guessed at.

Outputs (next to this script, in data/validation/):
- senegal_inventory_vs_powerplants.xlsx with three sheets:
    * "matched"            one row per inventory unit with its best match
    * "inventory_unmatched" inventory units with no acceptable match
    * "powerplants_unmatched" powerplants.csv sites with no inventory match
- A short summary is printed to stdout (total MW per source, matched vs.
  unmatched MW, per-fuel-type totals).

Run with:  python3 compare_senegal_inventory.py
"""

from __future__ import annotations

import difflib
import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
INVENTORY_PATH = ROOT / "data" / "validation" / "Supplementary-Data-1_INVENTORY.xlsx"
POWERPLANTS_PATH = ROOT / "resources" / "powerplants.csv"
OUTPUT_PATH = ROOT / "data" / "validation" / "senegal_inventory_vs_powerplants.xlsx"

# Matching thresholds - tuned by eye against the known Senegal fleet.
# Feel free to loosen/tighten these if you spot obviously wrong matches.
MAX_MATCH_DISTANCE_KM = 15.0      # distance beyond which proximity gives no credit
MATCH_SCORE_THRESHOLD = 0.35      # combined score needed to accept a match
NAME_WEIGHT = 0.6
PROXIMITY_WEIGHT = 0.4

# Words that show up in one source's naming convention but carry no
# identifying information, stripped before comparing names.
NOISE_WORDS = {
    "FARM", "SOLAR", "PV", "WIND", "TURBINE", "PLANT", "POWER", "STATION",
    "ENGINE", "TIMEPOINT", "CEMENT", "GOLD", "MINE", "AGGREKO", "HR",
    "SA", "SUGAR", "ACID", "PARK", "REGION", "SENELEC", "C1", "C2", "C3",
    "C4", "C5", "C6", "I", "II", "HFO", "TOWER",
}


def normalize_name(name: str) -> set[str]:
    """Upper-case, strip punctuation, drop noise words -> a token set."""
    if not isinstance(name, str):
        return set()
    cleaned = re.sub(r"[^A-Z0-9 ]", " ", name.upper())
    tokens = [t for t in cleaned.split() if t and t not in NOISE_WORDS]
    return set(tokens)


def name_similarity(name_a: str, name_b: str) -> float:
    """Blend of token-set Jaccard overlap and a character-level ratio."""
    tokens_a, tokens_b = normalize_name(name_a), normalize_name(name_b)
    if tokens_a and tokens_b:
        jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    else:
        jaccard = 0.0
    char_ratio = difflib.SequenceMatcher(
        None, str(name_a).upper(), str(name_b).upper()
    ).ratio()
    return 0.6 * jaccard + 0.4 * char_ratio


def haversine_km(lat1, lon1, lat2, lon2) -> float | None:
    if any(pd.isna(v) for v in (lat1, lon1, lat2, lon2)):
        return None
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_inventory_senegal() -> pd.DataFrame:
    df = pd.read_excel(INVENTORY_PATH, sheet_name="Main_Inventory")
    sen = df[df["Country"].astype(str).str.upper() == "SENEGAL"].copy()
    sen = sen.rename(
        columns={
            "Plant": "inv_name",
            "MW": "inv_mw",
            "GEN_TYPE": "inv_fueltype",
            "Operation": "inv_operation",
            "Specific Fuel": "inv_specific_fuel",
            "Operation Year": "inv_operation_year",
            "Latitude": "inv_lat",
            "Longitude": "inv_lon",
        }
    )
    return sen.reset_index(drop=True)


def load_powerplants() -> pd.DataFrame:
    df = pd.read_csv(POWERPLANTS_PATH)
    df = df.rename(
        columns={
            "Name": "pp_name",
            "Capacity": "pp_mw",
            "Fueltype": "pp_fueltype",
            "Technology": "pp_technology",
            "DateIn": "pp_date_in",
            "lat": "pp_lat",
            "lon": "pp_lon",
        }
    )
    return df.reset_index(drop=True)


def best_match(inv_row, pp_df: pd.DataFrame) -> tuple[int | None, float, float | None, float]:
    """Return (pp_index, combined_score, distance_km, name_score) of the
    best-scoring powerplants.csv row for one inventory row, or
    (None, 0, None, 0) if nothing clears the threshold."""
    best_idx, best_score, best_dist, best_name_score = None, 0.0, None, 0.0
    for pp_idx, pp_row in pp_df.iterrows():
        n_score = name_similarity(inv_row["inv_name"], pp_row["pp_name"])
        dist = haversine_km(
            inv_row["inv_lat"], inv_row["inv_lon"], pp_row["pp_lat"], pp_row["pp_lon"]
        )
        proximity_score = 0.0 if dist is None else max(0.0, 1 - dist / MAX_MATCH_DISTANCE_KM)
        combined = NAME_WEIGHT * n_score + PROXIMITY_WEIGHT * proximity_score
        if combined > best_score:
            best_idx, best_score, best_dist, best_name_score = pp_idx, combined, dist, n_score
    if best_score < MATCH_SCORE_THRESHOLD:
        return None, best_score, best_dist, best_name_score
    return best_idx, best_score, best_dist, best_name_score


def main() -> None:
    inv = load_inventory_senegal()
    pp = load_powerplants()

    matches = []
    matched_pp_indices = set()
    for _, inv_row in inv.iterrows():
        pp_idx, score, dist, name_score = best_match(inv_row, pp)
        record = {
            "inventory_plant": inv_row["inv_name"],
            "inventory_mw": inv_row["inv_mw"],
            "inventory_fueltype": inv_row["inv_fueltype"],
            "inventory_year": inv_row["inv_operation_year"],
            "match_score": round(score, 3),
            "distance_km": None if dist is None else round(dist, 2),
        }
        if pp_idx is not None:
            matched_pp_indices.add(pp_idx)
            pp_row = pp.loc[pp_idx]
            record.update(
                {
                    "matched_powerplant": pp_row["pp_name"],
                    "powerplant_mw": pp_row["pp_mw"],
                    "powerplant_fueltype": pp_row["pp_fueltype"],
                    "mw_diff_inv_minus_pp": round(inv_row["inv_mw"] - pp_row["pp_mw"], 2),
                }
            )
        else:
            record.update(
                {
                    "matched_powerplant": None,
                    "powerplant_mw": None,
                    "powerplant_fueltype": None,
                    "mw_diff_inv_minus_pp": None,
                }
            )
        matches.append(record)

    matches_df = pd.DataFrame(matches)
    inventory_unmatched = matches_df[matches_df["matched_powerplant"].isna()].copy()
    matched_only = matches_df[matches_df["matched_powerplant"].notna()].copy()
    powerplants_unmatched = pp.loc[~pp.index.isin(matched_pp_indices)].copy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_PATH) as writer:
        matched_only.sort_values("matched_powerplant").to_excel(
            writer, sheet_name="matched", index=False
        )
        inventory_unmatched.to_excel(writer, sheet_name="inventory_unmatched", index=False)
        powerplants_unmatched.to_excel(writer, sheet_name="powerplants_unmatched", index=False)

    # ---- summary printed to stdout ----
    inv_total = inv["inv_mw"].sum()
    pp_total = pp["pp_mw"].sum()
    matched_inv_total = matched_only["inventory_mw"].sum()
    unmatched_inv_total = inventory_unmatched["inventory_mw"].sum()
    unmatched_pp_total = powerplants_unmatched["pp_mw"].sum()

    print("=" * 70)
    print("SENEGAL: inventory vs. resources/powerplants.csv")
    print("=" * 70)
    print(f"Inventory rows (Senegal):      {len(inv):>4d}   total {inv_total:>9.2f} MW")
    print(f"powerplants.csv rows:          {len(pp):>4d}   total {pp_total:>9.2f} MW")
    print("-" * 70)
    print(f"Inventory rows matched:        {len(matched_only):>4d}   total {matched_inv_total:>9.2f} MW")
    print(f"Inventory rows UNmatched:      {len(inventory_unmatched):>4d}   total {unmatched_inv_total:>9.2f} MW")
    print(f"powerplants.csv rows UNmatched:{len(powerplants_unmatched):>4d}   total {unmatched_pp_total:>9.2f} MW")
    print("-" * 70)
    print("Inventory MW by GEN_TYPE:")
    print(inv.groupby("inv_fueltype")["inv_mw"].sum().sort_values(ascending=False).to_string())
    print("-" * 70)
    print("powerplants.csv MW by Fueltype:")
    print(pp.groupby("pp_fueltype")["pp_mw"].sum().sort_values(ascending=False).to_string())
    print("-" * 70)
    print(f"Full comparison written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
