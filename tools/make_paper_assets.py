"""Paper figures, LaTeX tables and a key-numbers sheet from the committed benchmark results.

Reads research/benchmark_results_v4.json (and research/guard_sweep_v4.json if it exists);
nothing is re-run. Writes to paper_assets/:
  fig_forecast, fig_drops_split, fig_tradeoff, fig_family, fig_inference   (.pdf + 300-dpi .png)
  tables.tex       booktabs tables: forecast metrics, routing outcomes, paired bootstrap CIs
  KEY_NUMBERS.md   every headline number with its JSON path

  python tools/make_paper_assets.py [--results PATH] [--sweep PATH] [--out DIR]

Figures are IEEE single-column: exactly 3.5 in wide, 8-pt serif text, TrueType fonts embedded
(IEEE rejects Type 3), Okabe-Ito colours with one fixed colour and marker per strategy; each
forecast predictor shares the colour of the guarded strategy that routes on it.
"""
import argparse, json, os, sys
from statistics import mean, stdev

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter

RESULTS = os.path.join(ROOT, "research", "benchmark_results_v4.json")
SWEEP = os.path.join(ROOT, "research", "guard_sweep_v4.json")
OUT = os.path.join(ROOT, "paper_assets")
WIDTH_IN, DPI = 3.5, 300
FIGURES = ["fig_forecast", "fig_drops_split", "fig_tradeoff", "fig_family", "fig_inference"]

FAMILIES = ["uniform", "hotspot", "elephant_mice", "stride", "llm_training", "mixed"]
FAMILY_LABELS = dict(uniform="Uniform", hotspot="Hotspot", elephant_mice="Elephant/\nmice", stride="Stride",
                     llm_training="LLM\ntraining", mixed="Mixed")
STRATEGIES = ["ecmp", "reactive", "gnn", "guarded_gnn", "guarded_lstm", "guarded_persistence", "guarded_score"]
GUARDED = ["guarded_gnn", "guarded_lstm", "guarded_persistence", "guarded_score"]
PREDICTORS = ["gnn", "lstm_only", "logreg", "persistence_continuous", "persistence_binary"]
GUARD_BASELINES = ["guarded_persistence", "guarded_score", "guarded_lstm"]

# Okabe-Ito (colourblind-safe). (label, colour, marker)
BLUE, SKY, VERMILLION, GREEN, ORANGE, PURPLE, BLACK = (
    "#0072B2", "#56B4E9", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#000000")
STYLE = {
    "ecmp": ("ECMP", BLACK, "X"),
    "reactive": ("Reactive", ORANGE, "^"),
    "gnn": ("GNN, unguarded", PURPLE, "P"),
    "guarded_gnn": ("Guarded GNN", BLUE, "o"),
    "guarded_lstm": ("Guarded LSTM", SKY, "s"),
    "guarded_persistence": ("Guarded persistence", GREEN, "v"),
    "guarded_score": ("Guarded score", VERMILLION, "D"),
    "p:gnn": ("GNN (GCN-LSTM)", BLUE, "o"),
    "p:lstm_only": ("LSTM-only", SKY, "s"),
    "p:logreg": ("Per-link LogReg", ORANGE, "^"),
    "p:persistence_continuous": ("Persistence, continuous", VERMILLION, "D"),
    "p:persistence_binary": ("Persistence, binary", GREEN, "v"),
}
FABRIC_GREY, HOST_GREY = "#4D4D4D", "#C8C8C8"
MINUS = "−"


def apply_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "legend.title_fontsize": 8,
        "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
        "legend.frameon": False, "legend.handlelength": 1.4, "legend.columnspacing": 1.0,
        "hatch.linewidth": .4, "figure.dpi": DPI, "savefig.dpi": DPI,
        "figure.constrained_layout.use": True,
    })


def save(fig, out_dir, name):
    """PDF + PNG at exactly WIDTH_IN wide (no tight bbox, so a PNG is WIDTH_IN * DPI pixels)."""
    paths = []
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=DPI, metadata={"CreationDate": None} if ext == "pdf" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def at(j, path):
    """Value at a dotted JSON path ('a.b.0.c'); raises if the path does not exist."""
    node = j
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def signed(x, digits=1, unicode_minus=True):
    s = f"{x:+.{digits}f}"
    return s.replace("-", MINUS) if unicode_minus else s


def check_inputs(j):
    m, o = j["forecast"]["metrics"], j["overall_summary"]
    assert set(PREDICTORS) == set(m), sorted(m)
    assert set(STRATEGIES) == set(o), sorted(o)
    assert all(o[s]["mean_unroutable_drops"] == 0 for s in STRATEGIES), "drop split must be fabric + host"
    assert {r["pattern"] for r in j["runs"]} == set(FAMILIES)


# ── Derived numbers ────────────────────────────────────────────────────────────

def family_fabric(j):
    """{(family, strategy): mean fabric drops per run}; model strategies averaged over seeds."""
    out = {}
    for fam in FAMILIES:
        for s in STRATEGIES:
            out[(fam, s)] = mean(r["fabric_link_drops"] for r in j["runs"]
                                 if r["pattern"] == fam and r["strategy"] == s)
    return out


def family_fabric_reduction(j):
    """{(family, strategy): (reduction %, std across training seeds or None)}; None where ECMP
    has no fabric drops in that family (reduction undefined)."""
    out = {}
    for fam in FAMILIES:
        runs = [r for r in j["runs"] if r["pattern"] == fam]
        ecmp = mean(r["fabric_link_drops"] for r in runs if r["strategy"] == "ecmp")
        for s in GUARDED:
            rs = [r for r in runs if r["strategy"] == s]
            if ecmp == 0:
                out[(fam, s)] = None
                continue
            seeds = sorted({r["train_seed"] for r in rs}, key=str)
            per_seed = [100 * (1 - mean(r["fabric_link_drops"] for r in rs if r["train_seed"] == t) / ecmp)
                        for t in seeds]
            out[(fam, s)] = (mean(per_seed), stdev(per_seed) if len(per_seed) > 1 else None)
    return out


def total_change_pct(j, s):
    return -at(j, f"vs_ecmp.{s}.total_drop_reduction_percent")


def stride_flow_sets(j):
    """Regenerate (no simulation) the stride traffic of every benchmark seed and return
    (number of seeds, number of distinct (flow ID, src, dst, ECMP path) sets, flows per seed)."""
    from simulation.topology import build_fat_tree
    from simulation.traffic_generator import TrafficGenerator
    from simulation.network_engine import NetworkEngine
    graph = build_fat_tree()
    engine = NetworkEngine(graph)
    seeds = sorted({r["seed"] for r in j["runs"] if r["pattern"] == "stride"})
    sets = set()
    for seed in seeds:
        flows = TrafficGenerator(graph, seed=seed).generate_scenario(total_ticks=j["settings"]["ticks"], pattern="stride")
        sets.add(tuple(sorted((f.flow_id, f.src, f.dst, tuple(engine.get_ecmp_path(f))) for f in flows)))
    return len(seeds), len(sets), len(next(iter(sets)))


# ── Figures ────────────────────────────────────────────────────────────────────

def fig_forecast(j):
    m = j["forecast"]["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 1.6), sharey=True)
    ys = list(range(len(PREDICTORS)))[::-1]
    for ax, part, title in zip(axes, ("overall", "onset"), ("(a) Overall AP", "(b) Onset AP")):
        for y, p in zip(ys, PREDICTORS):
            stat = m[p]["summary"][part]["average_precision"]
            label, color, marker = STYLE["p:" + p]
            xerr = stat["std"] if len(m[p]["per_seed"]) > 1 else None
            ax.errorbar(stat["mean"], y, xerr=xerr, fmt=marker, color=color, ms=4, elinewidth=.8, capsize=2)
        chance = m["gnn"]["summary"][part]["positive_rate"]["mean"]  # AP of a random ranking
        if part == "onset":  # the overall chance level (0.23) is far left of panel (a)'s range
            ax.axvline(chance, color=BLACK, ls="--", lw=.6)
        ax.text(.98, .02, f"chance = {chance:.3f}", transform=ax.transAxes, ha="right", va="bottom")
        ax.set_title(title)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.grid(axis="x", lw=.3, color="#DDDDDD")
    axes[0].set_xlim(.78, .97)
    axes[1].set_xlim(0, .22)
    axes[0].set_yticks(ys, [STYLE["p:" + p][0] for p in PREDICTORS])
    axes[0].set_ylim(-.6, len(PREDICTORS) - .4)
    return fig


def fig_drops_split(j):
    o, vs, spread = j["overall_summary"], j["vs_ecmp"], j["routing_seed_spread"]
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.15))
    ys = list(range(len(STRATEGIES)))[::-1]
    for y, s in zip(ys, STRATEGIES):
        fabric, host = o[s]["mean_fabric_link_drops"] / 1e3, o[s]["mean_host_link_drops"] / 1e3
        ax.barh(y, fabric, .62, color=FABRIC_GREY)
        ax.barh(y, host, .62, left=fabric, color=HOST_GREY, hatch="////", edgecolor="#8C8C8C", linewidth=0)
        total = fabric + host
        if s in spread:
            ax.errorbar(total, y, xerr=spread[s]["total_drops"]["std"] / 1e3, fmt="none", ecolor=BLACK,
                        elinewidth=.7, capsize=2)
        note = "reference" if s == "ecmp" else f"fabric {signed(-vs[s]['fabric_drop_reduction_percent'])}%"
        ax.text(total + 5, y, note, va="center")
    ax.set_yticks(ys, [STYLE[s][0] for s in STRATEGIES])
    ax.set_xlim(0, 330)  # room for the labels right of the bars
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.spines["bottom"].set_bounds(0, 200)
    ax.set_xlabel("Mean drops per run (thousands of packets)")
    ax.tick_params(axis="y", length=0)
    handles = [Patch(color=FABRIC_GREY, label="Fabric links"),
               Patch(facecolor=HOST_GREY, hatch="////", edgecolor="#8C8C8C", linewidth=0, label="Host links")]
    fig.legend(handles=handles, loc="outside upper center", ncol=2)
    return fig


def fig_tradeoff(j, sweep=None):
    o, spread = j["overall_summary"], j["routing_seed_spread"]
    ecmp = o["ecmp"]["mean_total_drops"]
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.75))
    ax.axhline(0, color=BLACK, ls="--", lw=.8)
    plotted = []  # x values, for the log-axis range
    for s in STRATEGIES[1:]:
        label, color, marker = STYLE[s]
        xerr = yerr = None
        if s in spread:
            xerr = spread[s]["reroutes"]["std"]
            yerr = spread[s]["total_drops"]["std"] / ecmp * 100
        x = o[s]["mean_reroutes_per_run"]
        plotted.append(x)
        ax.errorbar(x, total_change_pct(j, s), xerr=xerr, yerr=yerr, fmt=marker,
                    color=color, ms=4.5, elinewidth=.7, capsize=2, label=label)
    if sweep is not None:
        points = sweep["summary"]["max_changes"]
        xs = [p["mean_reroutes_per_run"] for p in points]
        ys = [p["total_change_vs_ecmp_percent"] for p in points]
        mcs = [p["max_changes"] for p in points]
        seeds = sweep["settings"]["traffic_seeds"]
        plotted += xs
        ax.plot(xs, ys, ls=":", marker="D", ms=4, mfc="white", color=VERMILLION,
                label=f"Sensitivity analysis: guarded score,\nmax_changes {mcs[0]}–{mcs[-1]}, "
                      f"seeds {seeds[0]}–{seeds[-1]}")
        for mc, x, y in zip(mcs, xs, ys):
            ax.annotate(str(mc), (x, y), textcoords="offset points", xytext=(0, 4.5), ha="center")
    ax.set_xscale("log")
    x_lo, x_hi = min(plotted) / 1.6, max(plotted) * 1.6
    ax.set_xlim(x_lo, x_hi)
    ax.xaxis.set_major_locator(FixedLocator([t for t in (5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
                                             if x_lo <= t <= x_hi]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Mean reroutes per run (log scale)")
    ax.set_ylabel("Total drops vs ECMP,\nsame traffic (%)")
    ax.set_ylim(min(ax.get_ylim()[0], -6.5), 1.2)
    ax.text(x_lo * 1.12, .25, "ECMP (0 reroutes)", va="bottom")
    ax.grid(lw=.3, color="#DDDDDD")
    fig.legend(loc="outside lower center", ncol=2)
    return fig


def fig_family(j):
    red, fabric = family_fabric_reduction(j), family_fabric(j)
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.35))
    n, width = len(GUARDED), .8 / len(GUARDED)
    for i, s in enumerate(GUARDED):
        label, color, _ = STYLE[s]
        xs, ys = [], []
        for k, fam in enumerate(FAMILIES):
            v = red[(fam, s)]
            if v is None:
                continue
            x = k + (i - (n - 1) / 2) * width
            xs.append(x); ys.append(v[0])
            if v[1] is not None:
                ax.errorbar(x, v[0], yerr=v[1], fmt="none", ecolor=BLACK, elinewidth=.6, capsize=1.5)
        ax.bar(xs, ys, width, color=color, label=label)
    for k, fam in enumerate(FAMILIES):
        if red[(fam, GUARDED[0])] is None:
            none_anywhere = all(fabric[(fam, s)] == 0 for s in STRATEGIES)
            ax.text(k, 50, "no fabric drops\nunder any strategy" if none_anywhere else "no fabric drops\nunder ECMP",
                    ha="center", va="center", rotation=90, color="#555555")
    ax.set_xticks(range(len(FAMILIES)), [FAMILY_LABELS[f] for f in FAMILIES])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Fabric-drop reduction\nvs ECMP (%)")
    ax.grid(axis="y", lw=.3, color="#DDDDDD")
    ax.set_axisbelow(True)
    fig.legend(loc="outside upper center", ncol=2)
    return fig


def fig_inference(j):
    t = j["inference_timing"]
    horizon = t["gnn"]["horizon_ms"]
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 1.5))
    models = [("gnn", "GNN (GCN-LSTM)", BLUE), ("lstm_only", "LSTM-only", SKY)]
    ys = list(range(len(models)))[::-1]
    for y, (k, label, color) in zip(ys, models):
        p50, p99 = t[k]["p50_ms"], t[k]["p99_ms"]
        ax.plot([p50, p99], [y, y], color=color, lw=1.6, zorder=1)
        ax.scatter([p50], [y], s=24, facecolor="white", edgecolor=color, linewidth=1.2, zorder=2)
        ax.scatter([p99], [y], s=24, color=color, zorder=2)
        ax.text(horizon + .15, y, f"{100 * t[k]['share_over_horizon']:.2f}% of calls > {horizon:g} ms", va="center")
    ax.axvline(horizon, color=BLACK, ls="--", lw=.8)
    ax.text(horizon - .08, len(models) - .45, f"{horizon:g} ms forecast horizon", ha="right", va="center")
    ax.set_yticks(ys, [m[1] for m in models])
    ax.set_ylim(-.6, len(models) - .2)
    ax.set_xlim(0, 8.6)  # room for the labels right of the horizon
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.spines["bottom"].set_bounds(0, 5)
    ax.set_xlabel("Per-tick inference time (ms)")
    ax.tick_params(axis="y", length=0)
    handles = [Line2D([], [], ls="", marker="o", mfc="white", mec=BLACK, label="p50"),
               Line2D([], [], ls="", marker="o", color=BLACK, label="p99")]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, handletextpad=.2)
    return fig


# ── LaTeX tables ───────────────────────────────────────────────────────────────

def tex_int(x):
    return f"{x:,.0f}".replace(",", "{,}")


def tex_signed_int(x):
    return f"${'-' if x < 0 else '+'}{tex_int(abs(x))}$"


def tex_signed(x, digits=2):
    return f"${x:+.{digits}f}$"


def stat_cell(stat, seeds, digits=3, dagger=False):
    """'0.947\\,$\\pm$\\,.003' (std without its leading zero, so 10 columns fit a two-column page)."""
    if not isinstance(stat, dict) or stat.get("mean") is None:
        return "---"
    cell = f"{stat['mean']:.{digits}f}"
    if seeds > 1:
        std = f"{stat['std']:.{digits}f}"
        cell += f"\\,$\\pm$\\,{std[1:] if std.startswith('0.') else std}"
    return cell + ("$^\\dagger$" if dagger else "")


def table_forecast(j):
    m = j["forecast"]["metrics"]
    overall_rate = m["gnn"]["summary"]["overall"]["positive_rate"]["mean"]
    onset_rate = m["gnn"]["summary"]["onset"]["positive_rate"]["mean"]
    n_all, n_onset = m["gnn"]["summary"]["overall"]["n"]["mean"], m["gnn"]["summary"]["onset"]["n"]["mean"]
    seeds = len(j["forecast"]["training_seeds"])
    rows = []
    for p in PREDICTORS:
        s, k = m[p]["summary"], len(m[p]["per_seed"])
        cells = [STYLE["p:" + p][0], stat_cell(s["threshold"], k)]
        for part in ("overall", "onset"):
            cells += [stat_cell(s[part]["f1"], k), stat_cell(s[part]["average_precision"], k),
                      stat_cell(s[part]["roc_auc"], k),
                      stat_cell(s[part]["brier"], k, dagger=not m[p]["summary"]["brier_is_probability"])]
        rows.append(" & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.5pt}",
        rf"\caption{{Forecast quality on the test set ({tex_int(n_all)} link-windows; onset: the "
        rf"{tex_int(n_onset)} windows not congested at the last observed tick). Mean $\pm$ std over "
        rf"{seeds} training seeds for the learned models; single values are deterministic. Thresholds "
        r"were tuned on validation (max F1). $^\dagger$Computed on the clipped congestion score, which is "
        r"not a probability.}",
        r"\label{tab:forecast}",
        r"\begin{tabular}{lc cccc cccc}",
        r"\toprule",
        rf" & & \multicolumn{{4}}{{c}}{{Overall (positive rate {100 * overall_rate:.1f}\%)}} & "
        rf"\multicolumn{{4}}{{c}}{{Onset (positive rate {100 * onset_rate:.1f}\%)}} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
        r"Predictor & Threshold & F1 & AP & AUC & Brier & F1 & AP & AUC & Brier \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])


def table_routing(j):
    o, vs, spread = j["overall_summary"], j["vs_ecmp"], j["routing_seed_spread"]
    pairs = vs["guarded_gnn"]["matched_trials"]
    rows = []
    for s in STRATEGIES:
        r, sp = o[s], spread.get(s)
        pm = lambda key, std_key, f=tex_int: f(r[key]) + (f" $\\pm$ {f(sp[std_key]['std'])}" if sp else "")
        rows.append(" & ".join([
            STYLE[s][0],
            pm("mean_total_drops", "total_drops"),
            "---" if s == "ecmp" else tex_signed(total_change_pct(j, s)),
            pm("mean_fabric_link_drops", "fabric_link_drops"),
            pm("mean_host_link_drops", "host_link_drops"),
            tex_int(r["mean_delivered_packets_per_tick"]),
            f"{r['mean_delivered_latency_us'] / 1000:.2f}",
            f"{r['mean_run_p99_latency_us'] / 1000:.2f}",
            pm("mean_reroutes_per_run", "reroutes", f=lambda x: f"{x:.1f}"),
        ]) + r" \\")
    return "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{Routing outcomes: mean per 150-tick run over {pairs} paired runs (6 traffic families "
        r"$\times$ 10 traffic seeds). $\pm$ is the std across the 3 training seeds of each seed's mean "
        r"(model strategies only). Delay is the mean delivered-packet delay; p99 is the mean of the "
        r"per-run 99th percentiles.}",
        r"\label{tab:routing}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Strategy & Total drops & vs ECMP (\%) & Fabric drops & Host drops & Delivered / tick & "
        r"Delay (ms) & p99 (ms) & Reroutes / run \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])


def paired_rows(j):
    return [p for b in GUARD_BASELINES for p in j["guarded_gnn_paired"]
            if p["a"] == "guarded_gnn" and p["b"] == b and p["key"] in ("total_drops", "fabric_link_drops")]


def table_paired(j):
    rows, last = [], None
    entries = paired_rows(j)
    for p in entries:
        name = STYLE[p["b"]][0] if p["b"] != last else ""
        if last is not None and p["b"] != last:
            rows.append(r"\addlinespace")
        last = p["b"]
        lo, hi = p["ci95"]
        rows.append(" & ".join([
            name, "total" if p["key"] == "total_drops" else "fabric",
            f"{tex_signed_int(p['mean_difference'])} [{tex_signed_int(lo)}, {tex_signed_int(hi)}]",
            tex_signed(p["relative_difference_percent"]),
            f"{p['wins']}/{p['ties']}/{p['losses']}",
        ]) + r" \\")
    excludes = [p for p in entries if p["ci95"][0] > 0 or p["ci95"][1] < 0]
    verdict = ("Every interval includes 0." if not excludes else
               "Intervals excluding 0: " + ", ".join(f"{STYLE[p['b']][0]} ({p['key']})" for p in excludes) + ".")
    draws = j["settings"]["bootstrap_draws"]
    return "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{Guarded GNN minus each baseline: mean difference in drops per run with a paired "
        rf"bootstrap 95\% CI ({entries[0]['pairs']} (family, traffic seed) pairs, seed-averaged drops, "
        rf"{tex_int(draws)} draws). Negative means guarded GNN dropped fewer packets. W/T/L counts pairs "
        rf"where guarded GNN had fewer/equal/more drops. {verdict}}}",
        r"\label{tab:paired}",
        r"\begin{tabular}{llrrc}",
        r"\toprule",
        r"Baseline & Drops & Mean diff.\ [95\% CI] & Rel.\ (\%) & W/T/L \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])


def tables_tex(j):
    header = ("% Generated by tools/make_paper_assets.py from research/benchmark_results_v4.json; do not edit.\n"
              "% Requires \\usepackage{booktabs}. Tables (a) and (b) are table* (two-column width).\n")
    return header + "\n\n".join([table_forecast(j), table_routing(j), table_paired(j)]) + "\n"


# ── Key numbers ────────────────────────────────────────────────────────────────

def key_numbers_md(j, sweep=None):
    m = j["forecast"]["metrics"]
    P = lambda path: f"`{path}`"
    v = lambda path, fmt="{:.3f}": fmt.format(at(j, path))
    ap = lambda p, part: f"forecast.metrics.{p}.summary.{part}.average_precision"
    pm = lambda path, digits=3: f"{at(j, path + '.mean'):.{digits}f} ± {at(j, path + '.std'):.{digits}f}"
    boot = lambda name, part: f"forecast.bootstrap.{name}.{part}"
    ci = lambda path: "[{:+.3f}, {:+.3f}]".format(*at(j, path + ".ci95"))
    red = lambda s, k: f"vs_ecmp.{s}.{k}_drop_reduction_percent"
    fam_fabric = family_fabric(j)
    stride_ecmp = [r["fabric_link_drops"] for r in j["runs"] if r["pattern"] == "stride" and r["strategy"] == "ecmp"]
    stride_seeds, stride_sets, stride_flows = stride_flow_sets(j)
    if stride_sets == 1:
        stride_note = (
            f"- **Stride:** all {stride_seeds} traffic seeds generate the same {stride_flows} flows (same flow IDs, "
            "src and dst, hence the same ECMP paths; checked by regenerating them); only rates and timing differ. "
            f"ECMP's stride fabric drops ({min(stride_ecmp):,}–{max(stride_ecmp):,} per run, "
            "`runs[pattern=stride, strategy=ecmp]`) come from one fixed hash-collision pattern, which every "
            "guarded strategy removes. Do not count \"stride solved\" as independent results.")
    else:
        stride_note = (f"- **Stride:** the {stride_seeds} traffic seeds generate {stride_sets} distinct flow/ECMP-path "
                       "sets (checked by regenerating them).")
    llm_none = all(fam_fabric[("llm_training", s)] == 0 for s in STRATEGIES)
    llm_note = ("- **LLM training:** 0 fabric drops under every strategy (`runs[pattern=llm_training]`); all its "
                "drops are on host links, so no routing strategy changes it." if llm_none else
                f"- **LLM training:** {fam_fabric[('llm_training', 'ecmp')]:,.0f} mean fabric drops per run under ECMP "
                "(`runs[pattern=llm_training, strategy=ecmp]`).")
    temps = [t["temperature"] for t in j["forecast"]["training"]["gnn"]]
    brier_shift = max(abs(t["val_brier_raw"] - t["val_brier_calibrated"])
                      for name in ("gnn", "lstm_only") for t in j["forecast"]["training"][name])
    paired = paired_rows(j)
    rng = lambda xs, d=1: f"{min(xs):.{d}f}–{max(xs):.{d}f}"
    rerouting = STRATEGIES[1:]
    total_cut = [at(j, red(s, "total")) for s in rerouting]
    fabric_cut = [at(j, red(s, "fabric")) for s in rerouting]
    guard_total = [at(j, red(s, "total")) for s in GUARDED]
    guard_fabric = [at(j, red(s, "fabric")) for s in GUARDED]
    guard_reroutes = [at(j, f"overall_summary.{s}.mean_reroutes_per_run") for s in GUARDED]
    reactive_reroutes = at(j, "overall_summary.reactive.mean_reroutes_per_run")
    lines = [
        "# Key numbers for the paper",
        "",
        "Generated by `tools/make_paper_assets.py` from `research/benchmark_results_v4.json`; rerun the "
        "script instead of editing. Each number is followed by its JSON path. \"± x\" is the sample std "
        "across the 3 training seeds. Full context: `reports/PHASE3_REPORT.md`.",
        "",
        "## Setup",
        f"- Results schema {v('schema_version', '{}')}, checkpoint schema {v('checkpoint_schema_version', '{}')}, "
        f"label mode `{v('label_mode', '{}')}` ({P('schema_version')}, {P('label_mode')}).",
        f"- 30 scenarios (6 families × 5), {v('settings.ticks', '{}')} ticks each, stratified split "
        f"{len(j['forecast']['splits']['train'])}/{len(j['forecast']['splits']['val'])}/"
        f"{len(j['forecast']['splits']['test'])} ({P('forecast.splits')}).",
        f"- Training seeds {j['reproducibility']['training_seeds']} ({P('reproducibility.training_seeds')}).",
        f"- Forecast test set: {v('forecast.metrics.gnn.summary.overall.n.mean', '{:,.0f}')} link-windows, "
        f"{100 * at(j, 'forecast.metrics.gnn.summary.overall.positive_rate.mean'):.1f}% congested 5 ticks ahead; "
        f"onset subset {v('forecast.metrics.gnn.summary.onset.n.mean', '{:,.0f}')} windows, "
        f"{100 * at(j, 'forecast.metrics.gnn.summary.onset.positive_rate.mean'):.1f}% positive "
        f"({P('forecast.metrics.gnn.summary.{overall,onset}.{n,positive_rate}')}).",
        f"- Routing: {v('vs_ecmp.guarded_gnn.matched_trials', '{}')} paired runs per strategy (6 families × "
        f"10 traffic seeds from {v('reproducibility.routing_seed_start', '{}')}) ({P('vs_ecmp.<strategy>.matched_trials')}).",
        "",
        "## Forecasting (test set)",
        f"- Overall AP: GNN {pm(ap('gnn', 'overall'))}, LSTM-only {pm(ap('lstm_only', 'overall'))}, "
        f"LogReg {v(ap('logreg', 'overall') + '.mean')}, continuous persistence "
        f"{v(ap('persistence_continuous', 'overall') + '.mean')}, binary persistence "
        f"{v(ap('persistence_binary', 'overall') + '.mean')} ({P('forecast.metrics.<predictor>.summary.overall.average_precision')}).",
        f"- GNN − continuous persistence: {v(boot('gnn_minus_persistence_continuous', 'overall') + '.point', '{:+.3f}')} "
        f"AP, 95% CI {ci(boot('gnn_minus_persistence_continuous', 'overall'))} "
        f"({P(boot('gnn_minus_persistence_continuous', 'overall'))}). The learned models rank congestion better.",
        f"- **GNN − LSTM-only: {v(boot('gnn_minus_lstm_only', 'overall') + '.point', '{:+.3f}')} AP, CI "
        f"{ci(boot('gnn_minus_lstm_only', 'overall'))}; onset {v(boot('gnn_minus_lstm_only', 'onset') + '.point', '{:+.3f}')}, "
        f"CI {ci(boot('gnn_minus_lstm_only', 'onset'))}** ({P('forecast.bootstrap.gnn_minus_lstm_only')}). "
        "Both include 0: the graph convolution adds nothing measurable.",
        f"- Onset AP: GNN {pm(ap('gnn', 'onset'))}, LSTM-only {pm(ap('lstm_only', 'onset'))}, LogReg "
        f"{v(ap('logreg', 'onset') + '.mean')}, continuous persistence {v(ap('persistence_continuous', 'onset') + '.mean')} "
        f"({P('forecast.metrics.<predictor>.summary.onset.average_precision')}). Onsets remain largely unpredicted.",
        f"- At its calibrated threshold the GNN catches {100 * at(j, 'forecast.metrics.gnn.summary.onset.recall.mean'):.1f}% "
        f"of onsets at {100 * at(j, 'forecast.metrics.gnn.summary.onset.precision.mean'):.1f}% precision "
        f"({P('forecast.metrics.gnn.summary.onset.{recall,precision}')}).",
        f"- Calibration: GNN temperatures {', '.join(f'{t:.2f}' for t in temps)} "
        f"({P('forecast.training.gnn.<i>.temperature')})"
        f"{'; all < 1, i.e. slightly under-confident' if all(t < 1 for t in temps) else ''}. Temperature "
        f"scaling changes validation Brier by at most {brier_shift:.4f} "
        f"({P('forecast.training.<model>.<i>.val_brier_{raw,calibrated}')}).",
        "",
        "## Routing (mean per run vs ECMP on the same traffic)",
        f"- ECMP: {v('overall_summary.ecmp.mean_total_drops', '{:,.0f}')} drops per run = "
        f"{v('overall_summary.ecmp.mean_fabric_link_drops', '{:,.0f}')} fabric + "
        f"{v('overall_summary.ecmp.mean_host_link_drops', '{:,.0f}')} host ({P('overall_summary.ecmp')}).",
    ]
    for s in STRATEGIES[1:]:
        lines.append(
            f"- {STYLE[s][0]}: total {signed(-at(j, red(s, 'total')), 2, False)}%, fabric "
            f"{signed(-at(j, red(s, 'fabric')), 2, False)}%, host {signed(-at(j, red(s, 'host')), 2, False)}%, "
            f"{at(j, f'overall_summary.{s}.mean_reroutes_per_run'):.1f} reroutes/run, W/T/L vs ECMP "
            f"{at(j, f'vs_ecmp.{s}.wins')}/{at(j, f'vs_ecmp.{s}.ties')}/{at(j, f'vs_ecmp.{s}.losses')} "
            f"({P(f'vs_ecmp.{s}')}, {P(f'overall_summary.{s}.mean_reroutes_per_run')}).")
    lines += [
        "- Host-link drops rise under every rerouting strategy: traffic no longer dropped in the fabric "
        f"reaches the receiver's edge→host link and is dropped there, so {rng(fabric_cut)}% fabric cuts "
        f"become {rng(total_cut)}% total cuts.",
        "- **Guarded GNN vs guard-only baselines** (paired bootstrap, "
        f"{paired[0]['pairs']} pairs; negative = guarded GNN better; {P('guarded_gnn_paired[b=…, key=…]')}):",
    ]
    for p in paired:
        lines.append(f"  - vs {STYLE[p['b']][0]}, {'total' if p['key'] == 'total_drops' else 'fabric'} drops: "
                     f"{p['mean_difference']:+,.0f} [{p['ci95'][0]:+,.0f}, {p['ci95'][1]:+,.0f}], "
                     f"{p['relative_difference_percent']:+.2f}%, W/T/L {p['wins']}/{p['ties']}/{p['losses']}.")
    excludes = [p for p in paired if p["ci95"][0] > 0 or p["ci95"][1] < 0]
    lines += [
        "  - " + ("Every interval includes 0: the GNN does not beat the guard-only baselines." if not excludes
                  else "Intervals excluding 0: " + ", ".join(f"{p['b']}/{p['key']}" for p in excludes) + "."),
        "",
        "## Inference (per-tick, inference only; 2 CPU threads, batch 1)",
    ]
    for k, name in (("gnn", "GNN"), ("lstm_only", "LSTM-only")):
        t = f"inference_timing.{k}"
        lines.append(f"- {name}: p50 {at(j, t + '.p50_ms'):.2f} ms, p99 {at(j, t + '.p99_ms'):.2f} ms, "
                     f"{100 * at(j, t + '.share_over_horizon'):.2f}% of {at(j, t + '.n'):,} calls over the "
                     f"{at(j, t + '.horizon_ms'):g} ms horizon ({P(t)}). Excludes telemetry export and route installation.")
    lines += [
        "",
        "## Caveats to state in the paper",
        stride_note,
        llm_note,
        "- 6 test scenarios: forecast CIs describe scenario-to-scenario variation, not population significance.",
        "- Synthetic simulator, one k = 4 topology, no PFC/ECN/RDMA, no route-installation delay; guard "
        "settings (max_changes 4, headroom 0.9, min_improvement 0.15) were not tuned.",
    ]
    if sweep is not None:
        pts = sweep["summary"]["max_changes"]
        seeds = sweep["settings"]["traffic_seeds"]
        lines += ["", f"## Sensitivity analysis: guarded score, max_changes (`research/guard_sweep_v4.json`, "
                      f"new traffic seeds {seeds[0]}–{seeds[-1]})"]
        for p in pts:
            lines.append(f"- max_changes {p['max_changes']}: total {p['total_change_vs_ecmp_percent']:+.2f}%, "
                         f"fabric {p['fabric_change_vs_ecmp_percent']:+.2f}%, {p['mean_reroutes_per_run']:.1f} "
                         f"reroutes/run, W/T/L {p['wins']}/{p['ties']}/{p['losses']} vs ECMP "
                         f"(`summary.max_changes[max_changes={p['max_changes']}]`).")
    lines += [
        "",
        "## What the paper can and cannot claim",
        f"- Can: a capacity guard turns a simple congestion signal into a {rng(guard_total)}% total and "
        f"{rng(guard_fabric)}% fabric drop reduction vs ECMP, with {rng(guard_reroutes)} reroutes per run "
        f"against {reactive_reroutes:.0f} for reactive rerouting "
        f"({reactive_reroutes / max(guard_reroutes):.0f}–{reactive_reroutes / min(guard_reroutes):.0f}× fewer), "
        "and it works with a model-free congestion score (guarded score is the best guarded variant).",
        "- Can: learned per-link temporal models rank congestion better than persistence.",
        "- Cannot: that the GNN (graph convolution) improves forecasting over an LSTM, or that GNN forecasts "
        "reduce drops beyond guard-only baselines.",
    ]
    return "\n".join(lines) + "\n"


# ── Entry point ────────────────────────────────────────────────────────────────

def build_all(results_path=RESULTS, out_dir=OUT, sweep_path=SWEEP):
    """Write every asset to out_dir; returns the written paths. The sweep is optional."""
    with open(results_path) as fh:
        j = json.load(fh)
    sweep = None
    if sweep_path and os.path.exists(sweep_path):
        with open(sweep_path) as fh:
            sweep = json.load(fh)
    check_inputs(j)
    os.makedirs(out_dir, exist_ok=True)
    apply_style()
    written = []
    written += save(fig_forecast(j), out_dir, "fig_forecast")
    written += save(fig_drops_split(j), out_dir, "fig_drops_split")
    written += save(fig_tradeoff(j, sweep), out_dir, "fig_tradeoff")
    written += save(fig_family(j), out_dir, "fig_family")
    written += save(fig_inference(j), out_dir, "fig_inference")
    for name, text in (("tables.tex", tables_tex(j)), ("KEY_NUMBERS.md", key_numbers_md(j, sweep))):
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", default=RESULTS)
    parser.add_argument("--sweep", default=SWEEP, help="optional; used if the file exists")
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    for path in build_all(args.results, args.out, args.sweep):
        print(os.path.relpath(path, ROOT))


if __name__ == "__main__":
    sys.exit(main())
