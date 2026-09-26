"""O0: sha256 of the EBB fixed-origin predictions on Carparts and Auto with the
current code (scalar discount path). Run before and after a model-code edit;
the two hashes must match bit for bit.
Usage: py scripts/integrity/o0_scalar_hash.py <tag>
"""
from __future__ import annotations
import hashlib
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from data_loading import load_generic_long, train_eval_split_last_h
from models.mixture_pooling import mixture_group_labels
from models.tsb_hb import fit_tsb_hb, predict_tsb_hb

OUT = ROOT / "outputs" / date.today().isoformat()
tag = sys.argv[1]
parts = []
for panel, structure, w in (("carparts", "global", 0.90), ("raf", "mixture", 0.997)):
    init, ev = train_eval_split_last_h(load_generic_long(ROOT / f"data/{panel}_long.csv"), h={"carparts": 6, "raf": 12}[panel])
    labels = mixture_group_labels(init, k=0, fit_discount=w).labels if structure == "mixture" else None
    params = fit_tsb_hb(init, group_labels=labels, item_variance_mode="conjugate",
                        item_variance_shrink_strength=20.0, fit_discount=w, bootstrap_draws=20, bootstrap_seed=42)
    pred = predict_tsb_hb(params, ev, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9], include_hyper_uncertainty=True)
    parts.append(pred.sort_values(["unique_id", "ds"]).to_csv(index=False))
h = hashlib.sha256("".join(parts).encode()).hexdigest()
(OUT / f"scalar_hash_{tag}.txt").write_text(h + "\n", encoding="utf-8")
print(tag, h)
