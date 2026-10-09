"""Apply reviewed prose corrections to the NeuralFabric paper.
Usage: python apply_paper_fixes.py papers/neuralfabric_ieee.tex
Each OLD string must occur exactly once; otherwise the script stops and changes nothing."""
import sys
FIXES = [
# 1. CI upper bound (forecast.bootstrap.gnn_minus_persistence_continuous.overall.ci95[1] = 0.1315)
(r"(95\% CI [+0.040, +0.132])", r"(95\% CI [+0.040, +0.131])"),
# 2. Abstract: guard's real value = comparable drop reduction, ~1/10 route changes, lower delay
(r"Every guarded variant reduces fabric-link drops by about 20--25\% with few route changes, and the gain persists on unseen traffic seeds; however, model-free guarded routing performs as well as forecast-driven guarded routing. The study therefore identifies capacity-aware admission of route changes, rather than predictor complexity, as the decisive factor, and contributes a reproducible ablation methodology that separates forecast quality from routing benefit.",
 r"Every guarded variant reduces fabric-link drops by about 20--25\%, matching aggressive rerouting with roughly one-tenth of the route changes and lowering tail delay below static hashing, and the gain persists on unseen traffic seeds; however, model-free guarded routing performs as well as forecast-driven guarded routing. Capacity-aware admission of route changes, rather than predictor complexity, therefore emerges as the decisive factor."),
# 3. Total reduction range (vs_ecmp.*.total_drop_reduction_percent: 4.86--5.72) + delay/route-change cost paragraph
(r"The total reduction is smaller (about 5--6\%) because part of the traffic that is no longer dropped in the fabric is subsequently dropped on the destination host link, which has no alternative path; rerouting cannot relieve an overloaded receiver.",
 r"The total reduction is smaller (4.9--5.7\%) because part of the traffic that is no longer dropped in the fabric is subsequently dropped on the destination host link, which has no alternative path; rerouting cannot relieve an overloaded receiver." "\n\n"
 r"Table~\ref{tab:routing} also exposes the cost of aggressive rerouting. Unguarded GNN rerouting reduces total drops by 5.51\%, slightly more than the guarded variants, and reactive rerouting by 4.15\%, but both perform more than 540 route changes per run and raise delivered delay above that of ECMP (99th percentile 17.17 and 17.60~ms versus 15.47~ms). The guarded variants obtain comparable drop reductions with 38--59 route changes per run and are the only strategies that lower delivered delay relative to ECMP (mean 7.26--7.30~ms versus 7.70~ms; 99th percentile 14.04--14.28~ms)."),
# 4. Mixed-family claim (per-family means: reactive does NOT beat guarded in mixed; unguarded GNN does)
(r"In the elephant/mice and mixed families, the more aggressive reactive and unguarded strategies achieve fewer drops than any guarded variant, at the cost of many more route changes.",
 r"In the elephant/mice family, both reactive and unguarded GNN rerouting achieve fewer drops than any guarded variant, and in the mixed family unguarded GNN rerouting does so as well, in both cases at the cost of roughly ten times more route changes."),
# 5. Sweep shift (max per-family shift = 0.512, hotspot, M=8 vs M=4)
(r"changes the result by less than 0.51 percentage points in every family",
 r"changes the result by at most 0.51 percentage points in any family"),
# 6. Brier caveat for continuous persistence (brier_is_probability: false)
(r"\caption{Forecast quality on test scenarios (mean $\pm$ std over three training seeds where applicable).}",
 r"\caption{Forecast quality on test scenarios (mean $\pm$ std over three training seeds where applicable). The Brier score of continuous persistence is computed on the clipped congestion score, not on a probability.}"),
# 7. Key findings: add the stability/delay result
(r"and forecast-driven guarded routing does not outperform the same controller driven by the current congestion score.",
 r"and forecast-driven guarded routing does not outperform the same controller driven by the current congestion score. Guarded routing matches the drop reduction of aggressive rerouting with about one-tenth of the route changes and is the only approach that reduces delivered delay relative to ECMP."),
# 8. Interpretation: correct the 'guard produces the gain' claim
(r"\emph{Interpretation.} The capacity guard, rather than the forecaster, produces the routing gain.",
 r"\emph{Interpretation.} The capacity guard does not yield larger drop reductions than aggressive rerouting; its contribution is to obtain comparable reductions with an order of magnitude fewer route changes and with lower delay."),
# 9. Conclusion
(r"All guarded variants reduced fabric-link drops by about 20--25\% with few route changes, and the benefit persisted on unseen traffic.",
 r"All guarded variants reduced fabric-link drops by about 20--25\%, matched the drop reduction of aggressive rerouting with roughly one-tenth of the route changes, lowered delivered delay below that of static hashing, and retained their benefit on unseen traffic."),
]
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
for i, (old, new) in enumerate(FIXES, 1):
    n = s.count(old)
    if n != 1:
        sys.exit(f"Fix {i}: expected 1 match, found {n}. Nothing written.")
for old, new in FIXES:
    s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
print(f"Applied {len(FIXES)} fixes to {p}")
