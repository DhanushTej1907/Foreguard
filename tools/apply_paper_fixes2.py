"""Second round of reviewed prose corrections (2 fixes).
Usage: python apply_paper_fixes2.py papers/neuralfabric_ieee.tex"""
import sys
FIXES = [
# guarded_score (5.72%) beats unguarded GNN (5.51%); only 3 of 4 guarded variants are below it
(r"Unguarded GNN rerouting reduces total drops by 5.51\%, slightly more than the guarded variants, and reactive rerouting by 4.15\%,",
 r"Unguarded GNN rerouting reduces total drops by 5.51\%, more than three of the four guarded variants (only guarded score, at 5.72\%, is higher), and reactive rerouting by 4.15\%,"),
# per-family route-change ratios: elephant/mice 6.7-10.5x, mixed 44-67x
(r"in both cases at the cost of roughly ten times more route changes.",
 r"in both cases at the cost of far more route changes (roughly 7--10 times as many in elephant/mice and 44--67 times as many in mixed)."),
]
p = sys.argv[1]; s = open(p, encoding="utf-8").read()
for i, (old, new) in enumerate(FIXES, 1):
    if s.count(old) != 1: sys.exit(f"Fix {i}: expected 1 match, found {s.count(old)}. Nothing written.")
for old, new in FIXES: s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s); print(f"Applied {len(FIXES)} fixes to {p}")
