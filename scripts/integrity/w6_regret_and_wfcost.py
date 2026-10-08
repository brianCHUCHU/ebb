"""W6: (a) max-regret-to-panel-best tables for the fixed-origin and walk-forward
protocols, (b) walk-forward wall-clock table, (c) EBB online wall-clock timing
on the four small panels (initialize -> per-block predict + update).

Inputs (all previously scored by the shared pipeline functions):
  outputs/integrity_2026-09-05/rescored_tab_prob.csv     fixed-origin mean SPL
  outputs/2026-09-06/wf_all_panels_wide.csv               walk-forward mean SPL
  outputs/paper_rebuild/wf_online_retail.csv, wf_carparts.csv   baseline walls
  outputs/paper_rebuild/adaptive_conformal_wf.csv    OR ACI walls
  outputs/2026-09-05/wf_roster_{auto,raf}.csv, wf_roster_carparts_aci.csv
  outputs/2026-09-05/h4_run_meta.json (M5 component walls)
  outputs/2026-09-06/w1_run_meta_<panel>.json (TweedieGP wf walls)
  outputs/2026-09-05/efficiency_carparts.csv (TweedieGP M5 full fit 7617.9 s, for projection)

Outputs (outputs/<date>/): regret_fixed.csv, regret_wf.csv, tab_regret_main.tex,
  tab_regret_full.tex, wf_cost.csv, tab_wfcost.tex, ebb_wf_timing.csv, w6_run_meta.json
Usage: py scripts/integrity/w6_regret_and_wfcost.py
"""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

OUT = ROOT / "outputs" / date.today().isoformat()
OUT.mkdir(parents=True, exist_ok=True)
RB = ROOT / "outputs" / "paper_rebuild"
D0905 = ROOT / "outputs" / "2026-09-05"
D0906 = ROOT / "outputs" / "2026-09-06"
I0905 = ROOT / "outputs" / "integrity_2026-09-05"
PANELS = ["online_retail", "carparts", "auto", "raf", "m5"]
PN = {"online_retail": "OR", "carparts": "Carparts", "auto": "Auto", "raf": "RAF", "m5": "M5"}
NAME = {"EB-Hurdle": "EBB", "EBB": "EBB", "CP-CrostonClassic": "CP-Croston", "CP-CrostonSBA": "CP-SBA"}
ORDER = ["AutoARIMA", "AutoTheta", "Tweedie-GLM", "iETS", "CP-Croston", "CP-SBA", "CP-TSB", "CP-ADIDA",
         "CP-IMAPA", "ACI-ADIDA", "ACI-TSB", "TSB-tuned", "TweedieGP", "EBB", "ACI-EBB", "EBB-rule"]
LABEL = {"EBB": r"\EBB{}", "ACI-EBB": r"ACI-\EBB{}", "EBB-rule": r"\EBB{} (rule)"}
TG_M5_FULL_FIT_S = 7617.9      # efficiency_carparts.csv, 12 workers, fixed-origin full fit
M5_BLOCKS = 93


def regret_table(mean: pd.DataFrame) -> pd.DataFrame:
    """mean: index model, columns panel -> regret % to the non-degenerate best
    (the rule row is excluded from the best, since it equals one of the two rows it chooses from)."""
    best = mean.drop(index=["Zero", "EBB-rule"], errors="ignore").min(axis=0)
    reg = 100.0 * (mean - best) / best
    reg["n_panels"] = mean.notna().sum(axis=1)
    reg["max_regret"] = reg[[c for c in mean.columns]].max(axis=1)
    reg["worst_panel"] = reg[[c for c in mean.columns]].idxmax(axis=1)
    reg["within5"] = (reg[[c for c in mean.columns]] <= 5.0).sum(axis=1)
    return reg


def fmt_reg(v):
    return "---" if pd.isna(v) else f"{v:.1f}"


def main():
    t0 = time.perf_counter()
    # ---------- (a) regret ----------
    fx = pd.read_csv(I0905 / "rescored_tab_prob.csv")
    fx["model"] = fx["model"].map(lambda m: NAME.get(m, m))
    fx_mean = fx.pivot_table(index="model", columns="panel", values="spl_mean")[PANELS]
    wf = pd.read_csv(OUT / "wf_all_panels_wide.csv")
    wf["model"] = wf["model"].map(lambda m: NAME.get(m, m))
    wf_mean = wf.pivot_table(index="model", columns="panel", values="mean")[PANELS]
    rf, rw = regret_table(fx_mean), regret_table(wf_mean)
    rf.to_csv(OUT / "regret_fixed.csv"); rw.to_csv(OUT / "regret_wf.csv")
    print("=== fixed-origin regret (%) ==="); print(rf.round(2).to_string())
    print("=== walk-forward regret (%) ==="); print(rw.round(2).to_string())

    models = [m for m in ORDER if m in rf.index or m in rw.index]
    # main (compact): Method | fixed max (panel) | wf max (panel) | within-5% fixed/wf
    lines = []
    for m in models:
        if m == "Zero":
            continue
        if m == "EBB":
            lines.append(r"\midrule")
        cells = [LABEL.get(m, m)]
        for r in (rf, rw):
            if m in r.index:
                n = int(r.loc[m, "n_panels"])
                cells.append(f"{r.loc[m, 'max_regret']:.1f} ({PN[r.loc[m, 'worst_panel']]})" + ("" if n == 5 else f"$^{{{n}}}$"))
                cells.append(f"{int(r.loc[m, 'within5'])}/{n}")
            else:
                cells += ["---", "---"]
        lines.append(" & ".join(cells) + r"\\")
    (OUT / "tab_regret_main.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # full: per-panel regret for both protocols
    lines = []
    for m in models:
        if m == "EBB":
            lines.append(r"\midrule")
        cells = [LABEL.get(m, m)]
        for r in (rf, rw):
            for p in PANELS:
                cells.append(fmt_reg(r.loc[m, p]) if m in r.index else "---")
        lines.append(" & ".join(cells) + r"\\")
    (OUT / "tab_regret_full.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------- (c) EBB online timing on the four small panels ----------
    from data_loading import (load_generic_long, load_online_retail, preprocess_online_retail,
                              train_eval_split_fixed_origin, train_eval_split_last_h)
    from experiments.protocols import iter_walk_forward_frames
    from models.mixture_pooling import mixture_group_labels
    from models.eb_hurdle import initialize_online_eb_hurdle, predict_online_eb_hurdle, update_online_eb_hurdle
    CONFIG = {"online_retail": ("global", 0.95), "carparts": ("global", 0.90),
              "auto": ("global", 0.99), "raf": ("mixture", 0.997)}
    STEP = {"online_retail": 7, "carparts": 1, "auto": 1, "raf": 1}
    timing = []
    for p, (structure, w) in CONFIG.items():
        if p == "online_retail":
            df = preprocess_online_retail(load_online_retail(ROOT / "data/online_retail.csv"))
            init_set, eval_set = train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
        else:
            h = {"auto": 6, "carparts": 6, "raf": 12}[p]
            init_set, eval_set = train_eval_split_last_h(load_generic_long(ROOT / f"data/{p}_long.csv"), h=h)
        labels = mixture_group_labels(init_set, k=0, fit_discount=w).labels if structure == "mixture" else None
        ti = time.perf_counter()
        state = initialize_online_eb_hurdle(
            init_set, group_labels=labels, bootstrap_draws=0, bootstrap_seed=42,
            group_shrink_strength=0.0, dynamic_occurrence=False, occurrence_discount=1.0,
            item_variance_mode="conjugate", item_variance_shrink_strength=20.0, fit_discount=w)
        t_init = time.perf_counter() - ti
        t_pred = t_upd = 0.0; nb = 0
        for frame in iter_walk_forward_frames(init_set, eval_set, step_size=STEP[p]):
            nb += 1
            tb = time.perf_counter()
            predict_online_eb_hurdle(state, frame.target, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
                                  n_samples=2000, include_hyper_uncertainty=False)
            t_pred += time.perf_counter() - tb
            tb = time.perf_counter()
            state = update_online_eb_hurdle(state, frame.target)
            t_upd += time.perf_counter() - tb
        timing.append({"panel": p, "n_series": int(init_set["unique_id"].nunique()), "blocks": nb,
                       "init_s": round(t_init, 2), "predict_s": round(t_pred, 2), "update_s": round(t_upd, 2),
                       "total_s": round(t_init + t_pred + t_upd, 2)})
        print(f"[timing] {p}: {timing[-1]}", flush=True)
    tm = pd.DataFrame(timing).set_index("panel"); tm.to_csv(OUT / "ebb_wf_timing.csv")

    # ---------- (b) walk-forward wall-clock table ----------
    def walls_long(path, panel):
        d = pd.read_csv(path)
        if "wall_clock_sec" in d.columns:
            d = d.dropna(subset=["wall_clock_sec"]).drop_duplicates("model")[["model", "wall_clock_sec"]]
            d = d.rename(columns={"wall_clock_sec": "s"})
        else:
            d = d[d["metric"] == "wall_clock_s"][["model", "value"]].rename(columns={"value": "s"})
        d["model"] = d["model"].map(lambda m: NAME.get(m, m)); d["panel"] = panel
        return d
    parts = [walls_long(RB / "wf_online_retail.csv", "online_retail"),
             walls_long(RB / "adaptive_conformal_wf.csv", "online_retail"),
             walls_long(RB / "wf_carparts.csv", "carparts"),
             walls_long(D0905 / "wf_roster_carparts_aci.csv", "carparts"),
             walls_long(D0905 / "wf_roster_auto.csv", "auto"),
             walls_long(D0905 / "wf_roster_raf.csv", "raf")]
    h4 = json.loads((D0905 / "h4_run_meta.json").read_text(encoding="utf-8"))["component_walls_s"]
    parts.append(pd.DataFrame({"model": ["CP-ADIDA", "ACI-ADIDA", "EBB"], "panel": "m5",
                               "s": [h4["CP5"] / 5.0, h4["ACI"], h4["EBB"]]}))
    for p in ["carparts", "auto", "raf", "online_retail"]:
        meta = json.loads((D0906 / f"w1_run_meta_{p}.json").read_text(encoding="utf-8"))
        parts.append(pd.DataFrame({"model": ["TweedieGP"], "panel": [p], "s": [meta["wall_clock_s"]]}))
    for p in PANELS:   # ACI-EBB walk-forward walls (W8: predict + ACI + update, one core)
        for d in sorted((ROOT / "outputs").glob("2026-09-*"), reverse=True):
            f = d / f"wf_aci_ebb_{p}.csv"
            if f.exists():
                a = pd.read_csv(f)
                v = a[(a["model"] == "ACI-EBB") & (a["metric"] == "wall_clock_s")]["value"]
                parts.append(pd.DataFrame({"model": ["ACI-EBB"], "panel": [p], "s": [float(v.iloc[0])]}))
                break
    walls = pd.concat(parts, ignore_index=True).dropna(subset=["s"])
    walls = walls[walls["model"] != "Zero"]
    # EBB: use the re-timed init+predict+update totals for the four small panels
    for p in tm.index:
        walls = walls[~((walls["panel"] == p) & (walls["model"] == "EBB"))]
        walls = pd.concat([walls, pd.DataFrame({"model": ["EBB"], "panel": [p], "s": [tm.loc[p, "total_s"]]})])
    walls = walls.drop_duplicates(["panel", "model"], keep="last")
    for d in sorted((ROOT / "outputs").glob("2026-09-*"), reverse=True):   # EBB-rule wall = wall of the chosen variant
        f = d / "aci_selector.csv"
        if f.exists():
            sel = pd.read_csv(f).set_index("panel")["apply_aci"].to_dict()
            for p, apply in sel.items():
                src = "ACI-EBB" if bool(apply) else "EBB"
                v = walls[(walls["panel"] == p) & (walls["model"] == src)]["s"]
                if len(v):
                    walls = pd.concat([walls, pd.DataFrame({"model": ["EBB-rule"], "panel": [p], "s": [float(v.iloc[-1])]})])
            break
    wide = walls.drop_duplicates(["panel", "model"], keep="last").pivot_table(index="model", columns="panel", values="s")[PANELS]
    wide.loc["TweedieGP", "m5"] = np.nan
    wide.to_csv(OUT / "wf_cost.csv")
    print("=== wf wall-clock (s) ==="); print(wide.round(1).to_string())
    tg_m5_proj_4 = TG_M5_FULL_FIT_S * int(np.ceil(M5_BLOCKS / 4)) / 3600
    tg_m5_proj_1 = TG_M5_FULL_FIT_S * M5_BLOCKS / 3600
    print(f"TweedieGP M5 wf projection: {tg_m5_proj_4:.0f} h (refit every 4 blocks), {tg_m5_proj_1:.0f} h (every block)")
    rows_order = ["AutoARIMA", "AutoTheta", "iETS", "CP-ADIDA", "ACI-ADIDA", "TweedieGP", "EBB", "ACI-EBB", "EBB-rule"]
    lines = []
    for m in rows_order:
        if m not in wide.index:
            continue
        if m == "EBB":
            lines.append(r"\midrule")
        cells = [LABEL.get(m, m)]
        for p in PANELS:
            v = wide.loc[m, p]
            if m == "TweedieGP" and p == "m5":
                cells.append(rf"$\approx${tg_m5_proj_4:.0f}\,h$^{{\dagger}}$")
            elif pd.isna(v):
                cells.append("---")
            elif v >= 3600:
                cells.append(f"{v/3600:.1f}\\,h")
            elif v >= 100:
                cells.append(f"{v:.0f}")
            else:
                cells.append(f"{v:.1f}")
        lines.append(" & ".join(cells) + r"\\")
    (OUT / "tab_wfcost.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {"task": "w6_regret_and_wfcost", "regret": "100*(mean SPL - best non-degenerate)/best per panel; Zero excluded from best",
            "tweediegp_m5_projection_h": {"refit_every_4_blocks": round(tg_m5_proj_4, 1), "refit_every_block": round(tg_m5_proj_1, 1),
                                          "basis": f"{TG_M5_FULL_FIT_S} s full fit at 12 workers x ceil(93/4)=24 or 93 refits"},
            "ebb_timing": "initialize_online_eb_hurdle + per-block predict_online_eb_hurdle(2000 samples) + update_online_eb_hurdle; single process",
            "baseline_walls": "ARS_SF_NJOBS=1 single-process StatsForecast; TweedieGP 12 workers",
            "wall_clock_total_s": round(time.perf_counter() - t0, 1), "platform": platform.platform(),
            "python": sys.version.split()[0], "seed": 42}
    (OUT / "w6_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
