"""Sensitivity of the capacity guard to its per-tick change budget (max_changes), on new traffic.

guarded_score (the best guard-only strategy in the benchmark) at max_changes in {2, 4, 8, 16},
plus ECMP, on traffic seeds 4000-4009: disjoint from the benchmark's routing seeds (3000-3009)
and from the forecast scenarios (seeds 42-71). 6 families, 150 ticks, alert threshold = the
benchmark's validation-tuned continuous-persistence threshold. No model is involved.

  python tools/guard_sweep.py [--max-minutes 20] [--out research/guard_sweep_v4.json]

The first (family, seed) block is timed; if the projected total exceeds --max-minutes the
script stops before running the rest and writes nothing.
"""
import argparse, inspect, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from config import MODEL, CHECKPOINT_SCHEMA_VERSION, RESULTS_OUT, check_output_path
from research_benchmark import PATTERNS, run_case, paired_bootstrap
from routing.guarded import GuardedRerouter
from simulation.topology import build_fat_tree
from simulation.traffic_generator import TrafficGenerator

MAX_CHANGES = [2, 4, 8, 16]
SEEDS = list(range(4000, 4010))
OUT = os.path.join(ROOT, "research", "guard_sweep_v4.json")
KEYS = ["total_drops", "fabric_link_drops", "host_link_drops", "reroutes"]


def strategy_name(max_changes):
    return f"guarded_score_mc{max_changes}"


def summarize(runs, max_changes=MAX_CHANGES, draws=2000):
    """Per max_changes: mean drops/reroutes, change vs ECMP on the same (family, seed) pairs,
    W/T/L and a paired bootstrap vs ECMP; plus per-family total-drop change."""
    by = {(r["pattern"], r["seed"], r["strategy"]): r for r in runs}
    pairs = sorted({(r["pattern"], r["seed"]) for r in runs})
    mean_of = lambda strategy, key, ps=pairs: float(np.mean([by[(p, s, strategy)][key] for p, s in ps]))
    change = lambda new, old: (new / old - 1) * 100 if old else None
    ecmp = {f"mean_{k}": mean_of("ecmp", k) for k in KEYS}
    points = []
    for mc in max_changes:
        name = strategy_name(mc)
        diffs = np.array([by[(p, s, name)]["total_drops"] - by[(p, s, "ecmp")]["total_drops"] for p, s in pairs])
        per_family = {}
        for fam in dict.fromkeys(p for p, _ in pairs):
            ps = [(p, s) for p, s in pairs if p == fam]
            per_family[fam] = dict(total_change_vs_ecmp_percent=change(mean_of(name, "total_drops", ps),
                                                                       mean_of("ecmp", "total_drops", ps)),
                                   fabric_change_vs_ecmp_percent=change(mean_of(name, "fabric_link_drops", ps),
                                                                        mean_of("ecmp", "fabric_link_drops", ps)))
        points.append(dict(
            max_changes=mc,
            mean_total_drops=mean_of(name, "total_drops"),
            mean_fabric_link_drops=mean_of(name, "fabric_link_drops"),
            mean_host_link_drops=mean_of(name, "host_link_drops"),
            mean_reroutes_per_run=mean_of(name, "reroutes"),
            total_change_vs_ecmp_percent=change(mean_of(name, "total_drops"), ecmp["mean_total_drops"]),
            fabric_change_vs_ecmp_percent=change(mean_of(name, "fabric_link_drops"), ecmp["mean_fabric_link_drops"]),
            host_change_vs_ecmp_percent=change(mean_of(name, "host_link_drops"), ecmp["mean_host_link_drops"]),
            wins=int((diffs < 0).sum()), ties=int((diffs == 0).sum()), losses=int((diffs > 0).sum()),
            paired_vs_ecmp={key: paired_bootstrap(runs, name, "ecmp", key, draws=draws)
                            for key in ("total_drops", "fabric_link_drops")},
            per_family=per_family,
        ))
    return dict(pairs=len(pairs), ecmp=ecmp, max_changes=points)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--max-minutes", type=float, default=20)
    parser.add_argument("--ticks", type=int, default=150)
    parser.add_argument("--benchmark", default=RESULTS_OUT, help="source of the guarded_score threshold")
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    check_output_path(args.out)
    with open(args.benchmark) as fh:
        bench = json.load(fh)
    thresholds = {name: bench["forecast"]["metrics"][name]["per_seed"][0]["threshold"]
                  for name in ("persistence_binary", "persistence_continuous")}
    benchmark_seeds = {r["seed"] for r in bench["runs"]}
    assert not benchmark_seeds & set(SEEDS), "sweep seeds must be new traffic"

    graph = build_fat_tree()
    strategies = [("ecmp", None)] + [(strategy_name(mc), mc) for mc in MAX_CHANGES]
    blocks = [(p, s) for p in PATTERNS for s in SEEDS]
    runs, started = [], time.perf_counter()
    for i, (pattern, seed) in enumerate(blocks):
        flows = TrafficGenerator(graph, seed=seed).generate_scenario(total_ticks=args.ticks, pattern=pattern)
        for name, mc in strategies:
            result, _ = run_case(graph, flows, args.ticks, "ecmp" if mc is None else "guarded_score", None,
                                 thresholds, guard_kwargs=None if mc is None else dict(max_changes=mc))
            assert result["packet_conservation"] and result["drop_split_ok"], (pattern, seed, name)
            runs.append(dict(pattern=pattern, seed=seed, strategy=name, max_changes=mc, **result))
        if i == 0:
            projected = (time.perf_counter() - started) * len(blocks) / 60
            print(f"First block took {time.perf_counter() - started:.1f} s; projected total {projected:.1f} min", flush=True)
            if projected > args.max_minutes:
                print(f"Projected {projected:.1f} min > --max-minutes {args.max_minutes:g}; stopping, nothing written.")
                return 1
        print(f"Completed {pattern}, seed {seed}", flush=True)

    defaults = {k: p.default for k, p in inspect.signature(GuardedRerouter.__init__).parameters.items()
                if p.default is not inspect.Parameter.empty and k != "threshold"}
    output = dict(
        schema_version=1, checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION, label_mode=MODEL["label_mode"],
        description="guarded_score with different max_changes vs ECMP on new traffic seeds (sensitivity analysis)",
        settings=dict(families=PATTERNS, traffic_seeds=SEEDS, ticks=args.ticks, max_changes=MAX_CHANGES,
                      threshold=thresholds["persistence_continuous"],
                      threshold_source=f"{os.path.relpath(args.benchmark, ROOT)}: "
                                       "forecast.metrics.persistence_continuous.per_seed[0].threshold",
                      guard_defaults=defaults, benchmark_routing_seeds=sorted(benchmark_seeds)),
        wall_clock_seconds=time.perf_counter() - started,
        summary=summarize(runs),
        runs=runs,
    )
    with open(args.out, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"Wrote {os.path.relpath(args.out, ROOT)} in {output['wall_clock_seconds'] / 60:.1f} min")
    for p in output["summary"]["max_changes"]:
        print(f"  max_changes {p['max_changes']:>2}: total {p['total_change_vs_ecmp_percent']:+.2f}%, "
              f"fabric {p['fabric_change_vs_ecmp_percent']:+.2f}%, reroutes/run {p['mean_reroutes_per_run']:.1f}, "
              f"W/T/L {p['wins']}/{p['ties']}/{p['losses']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
