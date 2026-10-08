"""D1: export the five panels (initialization / evaluation windows under the paper protocols) as
plain CSVs for the DeepAR runner, which lives in a separate Python 3.8 environment.

Output: outputs/deepar/panel_<panel>_{init,eval}.csv   (unique_id, ds, y)
Usage : py scripts/integrity/d1_deepar_export.py [panel ...]
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

spec = importlib.util.spec_from_file_location("c1", ROOT / "scripts" / "integrity" / "c1_coldstart.py")
c1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c1)

OUT = ROOT / "outputs" / "deepar"
OUT.mkdir(parents=True, exist_ok=True)
PANELS = ["carparts", "auto", "raf", "online_retail", "m5"]


def main() -> None:
    for p in (sys.argv[1:] or PANELS):
        init, ev = c1.load_panel(p)
        for name, df in (("init", init), ("eval", ev)):
            d = df[["unique_id", "ds", "y"]].copy()
            d["unique_id"] = d["unique_id"].astype(str)
            d["ds"] = pd.to_datetime(d["ds"]).dt.strftime("%Y-%m-%d")
            d.to_csv(OUT / f"panel_{p}_{name}.csv", index=False)
        print(f"[d1] {p}: init {len(init)} rows / {init['unique_id'].nunique()} series, eval {len(ev)} rows",
              flush=True)


if __name__ == "__main__":
    main()
