"""Convert Auto / RAF (gluon-ts intermittent-datasets branch) and Carparts
(Monash .tsf, Zenodo 4656022) into the long format used by this repo:
columns [unique_id, ds, y], monthly frequency.

Protocol alignment with Damato, Azzimonti & Corani (IJF 2025), Table 1:
  Auto:     3,000 series, T=18, h=6  (full series = gluonts test split, len 24)
  RAF:      5,000 series, T=72, h=12 (full series = gluonts test split, len 84)
  Carparts: 2,503 series, T=45, h=6  (series without missing values, len 51)

Usage:  py -m tools.convert_intermittent_datasets --raw data/raw_new --out data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def convert_gluonts(test_json: Path, prefix: str) -> pd.DataFrame:
    entries = json.load(open(test_json, encoding="utf-8"))
    rows = []
    for i, e in enumerate(entries):
        start = pd.Timestamp(e["start"]).to_period("M").to_timestamp()
        target = np.asarray(e["target"], dtype=float)
        ds = pd.date_range(start, periods=len(target), freq="MS")
        uid = f"{prefix}_{i}"
        rows.append(pd.DataFrame({"unique_id": uid, "ds": ds, "y": target}))
    return pd.concat(rows, ignore_index=True)


def convert_carparts_tsf(tsf_path: Path) -> pd.DataFrame:
    rows = []
    in_data = False
    for line in open(tsf_path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("@data"):
            in_data = True
            continue
        if not in_data:
            continue
        name, start_str, values = line.split(":", 2)
        vals = values.split(",")
        if any(v == "?" for v in vals):
            continue  # keep only series without missing values (matches TweedieGP's 2,503)
        y = np.asarray([float(v) for v in vals], dtype=float)
        if not np.any(y == 0):
            continue  # intermittency filter (ADI > 1), as in Damato et al. (2025)
        start = pd.Timestamp(start_str.replace(" 00-00-00", "")).to_period("M").to_timestamp()
        ds = pd.date_range(start, periods=len(y), freq="MS")
        rows.append(pd.DataFrame({"unique_id": f"carparts_{name}", "ds": ds, "y": y}))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw_new"))
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()

    auto = convert_gluonts(args.raw / "intermittent_auto" / "test" / "data.json", "auto")
    auto.to_csv(args.out / "auto_long.csv", index=False)
    print(f"auto: {auto['unique_id'].nunique()} series, len {auto.groupby('unique_id').size().unique()}")

    raf = convert_gluonts(args.raw / "intermittent_raf" / "test" / "data.json", "raf")
    raf.to_csv(args.out / "raf_long.csv", index=False)
    print(f"raf: {raf['unique_id'].nunique()} series, len {raf.groupby('unique_id').size().unique()}")

    carparts = convert_carparts_tsf(args.raw / "carparts" / "car_parts_dataset_with_missing_values.tsf")
    carparts.to_csv(args.out / "carparts_long.csv", index=False)
    print(f"carparts: {carparts['unique_id'].nunique()} series, len {carparts.groupby('unique_id').size().unique()}")


if __name__ == "__main__":
    main()
