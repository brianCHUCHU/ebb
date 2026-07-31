"""Static sanity checks for paper_v2/main.tex (env balance, refs, cites)."""
import re
from collections import Counter
from pathlib import Path

src = Path(__file__).resolve().parents[2].joinpath("paper_v2", "main.tex").read_text(encoding="utf-8")

begins = re.findall(r"\\begin\{(\w+\*?)\}", src)
ends = re.findall(r"\\end\{(\w+\*?)\}", src)
cb, ce = Counter(begins), Counter(ends)
mismatch = {k: (cb[k], ce.get(k, 0)) for k in set(cb) | set(ce) if cb.get(k, 0) != ce.get(k, 0)}
print("env mismatch:", mismatch or "none")

labels = set(re.findall(r"\\label\{([^}]+)\}", src))
refs = set(re.findall(r"\\ref\{([^}]+)\}", src))
print("unresolved refs:", (refs - labels) or "none")

cites = set()
for m in re.findall(r"\\cite[tp]?\{([^}]+)\}", src):
    cites.update(x.strip() for x in m.split(","))
bibs = set(re.findall(r"\\bibitem\[[^\]]*\]\{([^}]+)\}", src))
print("missing bibitems:", (cites - bibs) or "none")
print("uncited bibitems:", (bibs - cites) or "none")

# crude brace balance inside table blocks
n_open, n_close = src.count("{"), src.count("}")
print(f"brace count: {{ {n_open} vs }} {n_close}")
print("dollar count even:", src.count("$") % 2 == 0)
