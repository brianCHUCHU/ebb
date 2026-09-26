"""Fix the truncated row terminator after the TweedieGP line of tab:efficiency
(a single backslash left by an earlier edit) in both manuscript directories."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for d in ("paper_v2/v5", "paper_v2/v5_aistats"):
    p = ROOT / d / "sections" / "experiment_appendix.tex"
    s = p.read_text(encoding="utf-8")
    bad = "Python/torch & No\\\n\\midrule"
    good = "Python/torch & No\\\\\n\\midrule"
    print(d, "occurrences:", s.count(bad))
    p.write_text(s.replace(bad, good), encoding="utf-8")
