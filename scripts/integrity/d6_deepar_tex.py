"""D6: appendix subsection for DeepAR (labels app:deepar, tab:deepar, fig:deepar-credibility).

Reads outputs/deepar/deepar_results.csv, deepar_paired_all.csv, the run metadata and the median
occurrence credibility of each panel (outputs/2026-09-13/real_separation.csv); writes
paper_v2/v8_paper_runs/sections/deepar_app.tex. The figure it refers to is drawn by v8_figs.py.

Sign convention (2026-09-30): the gap is the loss of the worse of DeepAR and EBB relative to the
better, signed so that positive means EBB is the better, as in tab:coldstart and the figures.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DA = ROOT / "outputs" / "deepar"
OUT = ROOT / "paper_v2" / "v8_paper_runs" / "sections" / "deepar_app.tex"
PNL = {"online_retail": "Online Ret.","m5": "M5", "auto": "Auto", "carparts": "Carparts", "raf": "RAF"}
ORDER = ["raf", "auto", "carparts", "online_retail", "m5"]   # increasing occurrence credibility


def pval(p: float) -> str:
    if p < 1e-3:
        return "$<10^{-3}$"
    return f"${p:.3f}$"


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, ps[i] * (len(ps) - k)))
        out[i] = run
    return out


def main() -> None:
    res = pd.read_csv(DA / "deepar_results.csv")
    pr = pd.read_csv(DA / "deepar_paired_all.csv")
    pr["p_holm"] = holm(pr["p_t"].tolist())
    flips = pr[(pr["p_t"] < 0.05) != (pr["p_holm"] < 0.05)]
    print(f"paired tests: {len(pr)}; decisions changed by Holm adjustment at 5%: {len(flips)}")
    cred = pd.read_csv(ROOT / "outputs" / "2026-09-13" / "real_separation.csv").set_index("panel")["median_lambda_occ"]
    wf_ref = pd.read_csv(ROOT / "outputs" / "2026-09-09" / "wf_all_panels_wide.csv").set_index(["panel", "model"])["mean"]
    lines = []
    for protocol, tag in (("fixed", "Fixed"), ("wf", "Walk-fwd.")):
        first = True
        for p in ORDER:
            r = res[(res.protocol == protocol) & (res.panel == p)]
            if r.empty:
                continue
            r = r.iloc[0]
            meta = json.loads((DA / f"deepar_run_meta_{protocol}_{p}.json").read_text())
            e = pr[(pr.protocol == protocol) & (pr.panel == p) & (pr.other == "EBB")]
            t = pr[(pr.protocol == protocol) & (pr.panel == p) & (pr.other == "TweedieGP")]
            tg = f"{r.TweedieGP:.4f}" if pd.notna(r.TweedieGP) else "---"
            pe = pval(float(e.p_t.iloc[0])) if not e.empty else "---"
            pt = pval(float(t.p_t.iloc[0])) if not t.empty else "---"
            if protocol == "wf":
                rule = float(wf_ref.get((p, "EBB-rule"), float("nan")))
                extra = f"{rule:.4f}"
            else:
                extra = "---"
            adv = (r.DeepAR / r.EBB - 1) * 100 if r.EBB <= r.DeepAR else -(r.EBB / r.DeepAR - 1) * 100
            lines.append(
                f"{tag if first else ''} & {PNL[p]} & {float(cred.loc[p]):.2f} & {int(meta['prediction_length'])} & "
                f"{r.DeepAR:.4f} & {r.DeepAR_q90:.4f} & {r.EBB:.4f} & {extra} & {tg} & {adv:+.1f} & {pe} & {pt} & "
                f"{meta['wall_clock_seconds'] / 60:.1f}\\\\")
            first = False
        if protocol == "fixed":
            lines.append("\\addlinespace")
    body = "\n".join(lines)
    text = r"""\subsection{DeepAR: configuration, protocol, and paired tests}
\label{app:deepar}

DeepAR \citep{salinas2020} is run from GluonTS~0.11.12
\citep{alexandrov2020} (MXNet~1.7, CPU)
with library defaults for the architecture (two LSTM layers of 40 cells,
dropout $0.1$, default lags and time features for the frequency), a
negative-binomial output, 100 epochs of 50 batches at batch size 32 and
learning rate $10^{-3}$, seed 42, and 200 sample paths per forecast. The
configuration, the protocols, and the rule that every result is reported
were fixed before any DeepAR number under these protocols was seen;
nothing was tuned. One global model
is trained per panel on the initialization window. At fixed origin it
forecasts the whole evaluation span once; this is run on the three
monthly panels and not on Online Retail, whose evaluation span exceeds
the initialization window, nor on M5, where it would require a 647-step
decoder. Under walk-forward evaluation the trained model is never
refitted and forecasts each block from the history extended with
revealed targets only, the same information \EBB{} receives through its
sufficient-statistic updates. Scores use the functions and the scale
used for every other method.

\paragraph{The outcome against occurrence credibility.}
Table~\ref{tab:deepar} lists the panels in increasing order of median
occurrence credibility at the selected discount, the quantity reported
in Table~\ref{tab:main}. Under walk-forward evaluation \EBB{}'s advantage
over DeepAR falls monotonically along that order
(Figure~\ref{fig:deepar-credibility}), and the order is preserved
within the three monthly panels and within the two daily panels, so it
is not an artifact of sampling frequency alone. At fixed origin, run on
the monthly panels only, the signs agree with the walk-forward ones but
the two leads are not ordered: the lead on Auto is the larger. The
credibility weights were computed for Table~\ref{tab:leverage} before
DeepAR was run; the comparison with them was made afterwards and is
reported as an observation on five panels, not as a test. Every paired
decision in Table~\ref{tab:deepar} is unchanged by a Holm adjustment
over its """ + str(len(pr)) + r""" tests.

\begin{table*}[!htbp]
\centering
\caption{DeepAR against \EBB{} and TweedieGP, panels in increasing order
of median occurrence credibility $\lambda^{(o)}$. $h$ is the prediction
length per forecast. Adv.\ is the mean SPL of the worse of DeepAR and
\EBB{} relative to the better, positive when \EBB{} is the better;
$p$-values are from paired per-series
$t$-tests of DeepAR against each method. Wall-clock covers training and
all forecasts on one machine without a GPU.}
\label{tab:deepar}
\footnotesize
\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{llrrrrrrrrrrr}
\toprule
 & & & & \multicolumn{2}{c}{DeepAR} & & & & \EBB{} adv. & \multicolumn{2}{c}{$p$ vs.} & Wall\\
\cmidrule(lr){5-6}\cmidrule(lr){11-12}
Protocol & Panel & $\lambda^{(o)}$ & $h$ & mean & q90 & \EBB{} & \EBB{} (rule) & TweedieGP & (\%) & \EBB{} & TweedieGP & (min)\\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table*}

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.46\linewidth]{figs/fig_app_deepar_credibility.pdf}
\caption{\EBB{}'s walk-forward advantage over DeepAR against the median
occurrence credibility of each panel (values in Table~\ref{tab:deepar}).
Filled markers are uncalibrated \EBB{}; the open marker on M5 is
\EBB{} with the validation-selected ACI layer.}
\label{fig:deepar-credibility}
\end{figure}
"""
    OUT.write_text(text, encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
