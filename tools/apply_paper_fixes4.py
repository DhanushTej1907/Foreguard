"""Round 4: fixes the item-2 mismatch, the unseen-seed scope nit, and the figure path.
Usage: python apply_paper_fixes4.py papers/neuralfabric_ieee.tex"""
import sys
FIXES = [
# Item 2: guarded variants are below unguarded GNN on fabric drops (all four) and total drops (three of four)
(r"The capacity guard yields drop reductions comparable to those of aggressive rerouting (larger than reactive rerouting, and below unguarded GNN rerouting only for fabric-link drops), but it does so with",
 r"The guarded variants reduce drops more than reactive rerouting and less than unguarded GNN rerouting (the only exception is guarded score on total drops), but they do so with"),
# Item 1 nit: only guarded score was run on the unseen seeds
(r"and the gain of model-free guarded routing persists on unseen traffic seeds;",
 r"and the gain of guarded routing driven by the current congestion score persists on unseen traffic seeds;"),
(r"the model-free guarded variant also retained its benefit on unseen traffic.",
 r"the guarded variant driven by the current congestion score also retained its benefit on unseen traffic."),
# Figures resolve whether the tex sits next to paper_assets/ (Overleaf) or in papers/ (repo)
(r"\usepackage{graphicx}",
 r"\usepackage{graphicx}" "\n" r"\graphicspath{{./}{../}}"),
]
p = sys.argv[1]; s = open(p, encoding="utf-8").read()
for i, (old, new) in enumerate(FIXES, 1):
    if s.count(old) != 1: sys.exit(f"Fix {i}: expected 1 match, found {s.count(old)}. Nothing written.")
for old, new in FIXES: s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s); print(f"Applied {len(FIXES)} replacements to {p}")
