"""Third round of reviewed prose corrections (8 replacements for the 6 findings).
Usage: python apply_paper_fixes3.py papers/neuralfabric_ieee.tex"""
import sys
FIXES = [
# 1a. Abstract: only guarded score was run on unseen seeds (guard_sweep_v4.json)
(r"lowering tail delay below static hashing, and the gain persists on unseen traffic seeds;",
 r"lowering tail delay below static hashing, and the gain of model-free guarded routing persists on unseen traffic seeds;"),
# 1b. Conclusion, same issue
(r"lowered delivered delay below that of static hashing, and retained their benefit on unseen traffic.",
 r"lowered delivered delay below that of static hashing; the model-free guarded variant also retained its benefit on unseen traffic."),
# 2. Guard vs aggressive rerouting: comparable, not 'not larger'
(r"The capacity guard does not yield larger drop reductions than aggressive rerouting; its contribution is to obtain comparable reductions with an order of magnitude fewer route changes and with lower delay.",
 r"The capacity guard yields drop reductions comparable to those of aggressive rerouting (larger than reactive rerouting, and below unguarded GNN rerouting only for fabric-link drops), but it does so with an order of magnitude fewer route changes and with lower delay."),
# 5. Herd-effect attribution was not isolated experimentally
(r"and the guard's sequential reservation prevents the herd effect that degrades unguarded rerouting.",
 r"and the guard's admission rules, which include sequential capacity reservation, are consistent with avoiding the herd effect that affects unguarded rerouting, although the contribution of each rule was not isolated."),
# 3. Unguarded GNN description (routing/rerouter.py ProactiveRerouter: fixed 0.7, keeps cooldown)
(r"Reactive rerouting moves flows whose current path crosses a fabric link with utilisation above 0.9, and unguarded GNN rerouting moves flows away from risky links without the capacity and stability checks.",
 r"Reactive rerouting moves flows whose current path crosses a fabric link with utilisation above 0.9. Unguarded GNN rerouting moves flows away from links whose predicted probability exceeds a fixed alert threshold of 0.7; it keeps the five-tick per-flow cooldown but has no capacity check, per-tick budget, path-stretch bound or minimum-improvement condition."),
# 4. Only M was varied
(r"the weaker performance on elephant/mice and mixed traffic must be attributed to other guard parameters, such as the headroom factor or the minimum risk improvement.",
 r"the weaker performance on elephant/mice and mixed traffic may instead stem from other guard parameters, such as the headroom factor or the minimum risk improvement, which were not varied in this study."),
# 6. Hardware wording (integrated Intel Iris Xe present, not used)
(r"without a Graphics Processing Unit (GPU)",
 r"using the Central Processing Unit (CPU) only (the integrated Intel Iris Xe graphics were not used)"),
(r"PyTorch~2.14.1 (Central Processing Unit (CPU) build)",
 r"PyTorch~2.14.1 (CPU build)"),
# Future work: individual guard-rule ablation (follows from finding 5)
(r"and guard parameters should be tuned per workload.",
 r"and guard parameters should be tuned per workload, with each admission rule ablated individually."),
]
p = sys.argv[1]; s = open(p, encoding="utf-8").read()
for i, (old, new) in enumerate(FIXES, 1):
    if s.count(old) != 1: sys.exit(f"Fix {i}: expected 1 match, found {s.count(old)}. Nothing written.")
for old, new in FIXES: s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s); print(f"Applied {len(FIXES)} replacements to {p}")
