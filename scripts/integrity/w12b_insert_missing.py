"""W12b: insert the frontier figure and the walk-forward significance table if
their labels are missing (W12 checked for the ref, not the label)."""
from __future__ import annotations
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w12_aci_ebb_tex import DIRS, FIG_FRONTIER, TAB_WFSIG  # noqa: E402

for d in DIRS:
    p = d / "sections" / "experiments.tex"
    s = p.read_text(encoding="utf-8")
    if "\\label{fig:frontier}" not in s:
        i = s.index("\\end{table}\n", s.index("\\label{tab:regret}")) + len("\\end{table}\n")
        s = s[:i] + FIG_FRONTIER + s[i:]
        p.write_text(s, encoding="utf-8"); print("figure inserted", d.name)
    p = d / "sections" / "experiment_appendix.tex"
    s = p.read_text(encoding="utf-8")
    if "\\label{tab:wf-significance}" not in s:
        m = re.compile(r"\\end\{table\*?\}\n").search(s, s.index("\\label{tab:spl-significance}"))
        s = s[:m.end()] + TAB_WFSIG + s[m.end():]
        p.write_text(s, encoding="utf-8"); print("wf-significance table inserted", d.name)
