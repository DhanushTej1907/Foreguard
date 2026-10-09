# Phase 1 report: simulator fixes

Branch `fix/phase1`, one commit on top of `6bd9777` ("Baseline: Final Version").
Date: 2026-10-08.

## Summary

The simulator now models full-duplex links: each direction of a physical link has its own
queue, capacity and buffer. Before, both directions shared one FIFO. This was the main
change, and it moves every routing number:

- Mean ECMP drops fall from 301,822 to 213,467 per run (−29%).
- The guard-only gains shrink slightly: persistence-fed 5.55%, utilization-fed 5.89%.
- Two thirds of all ECMP drops are on host links, where rerouting cannot help.

Forecasting gets easier for the trivial baseline. Binary persistence F1 rises from 0.812 to
0.884, and continuous persistence reaches AP 0.893 / AUC 0.972. Onsets fall from 19.3% to
12.0% of positives.

The checkpoint schema is now 3. The existing `checkpoints/best_model.pt` (schema 2) is rejected
on purpose, so **there is no GNN result on the new simulator yet**. Phase 2 must retrain before
any GNN claim.

Tests: **20 / 20 pass** (13 existing, 7 new).

---

## Changes by item

### a. Full-duplex links (#6)
- `simulation/network_engine.py`:
  - `link_states` holds two `LinkState`s per graph edge, keyed `(u, v)` and `(v, u)`: 96
    directed states for the 48 physical links.
  - `_edge_key` returns the directed pair; `_physical_key` returns the sorted pair.
  - Switch snapshot aggregates now cover **egress** ports only (output-queued model).
  - New helper `merge_directions(links)`: field-wise max over both directions per physical link.
- `simulation/telemetry.py`: edge features and labels come from `merge_directions`. The GNN still
  sees 32 links × 4 features; edge index and architecture are unchanged.
- `routing/guarded.py`: no logic change. It already used `engine._edge_key`, so demand, queue
  and headroom are now checked per direction. Risk uses the per-physical-link prediction.
- `routing/rerouter.py` (`ReactiveRerouter`): the congested-link set and path weights are now
  direction-aware, via a weight callable instead of copying the graph. `ProactiveRerouter`
  needs no change.
- `routing/comparator.py`: no change; checked by running it (see Verification).
- `dashboard/server.py`: the WebSocket message carries one merged entry per physical link (the
  frontend draws one line per link). The status scope string says "full-duplex".
- `tools/guard_ablation_check.py`:
  - Reductions are now measured against its own ECMP rerun.
  - Recorded columns are labelled `rec,v2` and flagged as not comparable.
  - Binary persistence uses the label at the last observed tick, so it follows `label_mode`.
  - Prints the drop split.

### b. Label mode (#9)
- `MODEL["label_mode"]` takes one of two values:
  - `"offered"` (default, unchanged behaviour): `packets_in / capacity > 0.85`.
  - `"served_or_queued"`: `packets_out / capacity > 0.85` OR `queue > 0.1 × buffer`.
    The 0.1 is `MODEL["queue_label_fraction"]`.
- A physical link is labelled congested if either direction meets the rule.
- An unknown mode raises `ValueError`.
- `model/train.py` records `label_mode` in the checkpoint, for provenance only.

### c. Reactive rerouting on switch-to-switch links only (#23)
- `ReactiveRerouter` ignores links that have a host endpoint. A congested host link has no
  alternative path, so it only produced useless reroutes.

### d. Drop split (#5)
- `engine.metrics()` adds `host_link_drops`, `fabric_link_drops`, `unroutable_drops` and
  `drop_split_ok`.
- The identity is `host + fabric + unroutable == total_drops`.
- `unroutable_drops` counts flows with no path, such as src == dst. The traffic generator never
  creates these, so it is 0 in every generated scenario.

### e. Config cleanup and docstring (#11, #12)
- Removed `SIMULATION["warmup_ticks"]`, `TRAFFIC["hotspot_probability"]` and
  `TRAINING["train_split" / "val_split" / "test_split"]`. None of them was referenced: splits
  use the defaults in `model/splits.py`.
- The `telemetry.py` docstring now reads "INT-style idealized telemetry": counters are read
  directly from simulator state, not carried in packet headers.

### f. Checkpoint schema version 3
- `config.CHECKPOINT_SCHEMA_VERSION = 3`.
- `model.gnn_model.check_checkpoint_schema()` is used by `main.py --evaluate` and the dashboard.
  `model/train.py` writes the constant.
- `research_benchmark.py`'s `schema_version=3` is the **results-JSON** schema, a separate counter,
  and a comment now says so.

### g. Tests
New tests in `tests/test_research.py`:

| Test | What it checks |
|---|---|
| `test_directional_queues_independent` | Opposing full-rate flows: separate state objects, 2 pkts/tick per direction, zero queue, zero drops |
| `test_conservation_full_duplex_fat_tree` | Hotspot scenario over 60 ticks: conservation holds every tick, 96 directed states |
| `test_drop_split_sums_to_total` | Host and fabric drops both > 0 and sum to the total; a src == dst flow lands in `unroutable_drops` |
| `TelemetryTests.test_max_over_directions` | Congestion in one direction reaches the features and label; shapes (32, 4) / (32,) |
| `TelemetryTests.test_label_modes` | A queue-only case is labelled 0 under `offered` and 1 under `served_or_queued`; an unknown mode raises |
| `ReactiveTests.test_switch_links_only` | A hot host link triggers no reroute; a hot fabric link does, and the new path avoids it |
| `CheckpointTests.test_schema_version` | Schema 2 and missing schema are rejected; schema 3 is accepted |

Updated tests:
- `ControllerTests.setup_case` maps both directions to the prediction index, as the collector
  does. Before, it relied on `g.edges()` order happening to match the direction of travel.
- `test_dashboard`: asserts `model_loaded` only when the on-disk checkpoint has the current
  schema; otherwise it asserts the static-ECMP fallback. As planned, the old assertion would
  have failed against the schema-2 checkpoint.

### h. CLAUDE.md
- Python version is now 3.14, with the expected torch.jit and PyG warnings.
- Documents the full-duplex engine, `label_mode`, the drop split and the schema constant.
- "Known findings" is marked as schema-2 results that need re-verification.

---

## Results on the new simulator

### Guard ablation (`python tools/guard_ablation_check.py`)
18 runs: 6 patterns × seeds 3000–3002, 150 ticks. Mean drops per run:

| | Old (schema 2) | New (Phase 1) |
|---|---|---|
| ECMP | 301,821.5 | **213,467.4** |
| Guarded + persistence labels | 283,618.3 (−6.03%) | **201,621.8 (−5.55%)** |
| Guarded + current utilization | 283,400.1 (−6.10%) | **200,884.2 (−5.89%)** |
| Guarded + GNN | 283,899.0 (−5.94%) | *not available, needs retrain* |

- `ECMP(rec,v2) == ECMP(mine)` is now **False**. This is expected: the recorded JSON comes from
  the old simulator.
- Stride is solved completely by the guard: 0 drops for both guard-only variants on all 3 seeds.
- llm_training is unchanged by any controller.

### ECMP drop split (18 runs combined)

| Host-link drops | Fabric-link drops | Unroutable | Total |
|---|---|---|---|
| 2,593,467 (67.5%) | 1,248,946 (32.5%) | 0 | 3,842,413 |

Most drops happen on host links, for example incast at a receiver's edge→host link. I did not
break this down per pattern. A host link is the only path to its host, so no rerouting
strategy can remove those drops. Phase 2 should report drop reduction on fabric links separately.

### Forecasting baselines on the 5 test scenarios (`label_mode = "offered"`)

| | Old | New |
|---|---|---|
| Test positive rate | 0.2075 | **0.1995** |
| Binary persistence F1 / AP / AUC | 0.812 / 0.699 / 0.880 | **0.884 / 0.805 / 0.926** |
| Continuous persistence AP / AUC | 0.825 / 0.951 | **0.893 / 0.972** |
| Samples not congested now | 17,294 | **17,452** |
| …of which congested at t+5 (onsets) | 870 | **520** |
| Onset share of positives | 0.193 | **0.120** |

The persistence bar the GNN has to beat is now higher, and onsets, where a forecaster could
add value, are rarer.

### Label-mode comparison (the same 5 test scenarios)
Scenarios 0 (uniform), 3 (stride), 8 and 20 (elephant_mice) and 23 (mixed); seeds 42+i,
150 ticks; 21,760 link-samples.

| | `offered` | `served_or_queued` |
|---|---|---|
| Positive rate | 0.1995 | 0.2084 |
| Positives | 4,342 | 4,535 |
| Not congested now | 17,452 | 17,283 |
| Onsets | 520 | 500 |
| Onset share of positives | 0.120 | 0.110 |
| Uniform | 0.0967 | 0.1027 |
| Stride | 0.1556 | 0.1558 |
| Elephant/mice | 0.1156 | 0.1191 |
| Mixed | 0.5142 | 0.5453 |

The two modes differ by less than one percentage point in positive rate. About 70% of the
193 extra positives under `served_or_queued` come from the one mixed scenario. That mode also
has slightly fewer onsets. I did not break down which condition (queue or served throughput)
produces the extra positives. In this simulator the choice of default is therefore a question
of semantics, not class balance:
- `served_or_queued` labels what the link experienced: backlog or served load.
- `offered` labels input demand, even in a tick where the buffer absorbs it.

This decision is left for Phase 2.

---

## Old checkpoint and data
- `checkpoints/best_model.pt` (schema 2) and `data/datasets.pkl` were **not modified**. They are
  tracked in git at `6bd9777`.
- The checkpoint is rejected with "Legacy checkpoint (schema 2, need 3): regenerate data and
  retrain". The dashboard falls back to static ECMP.
- `datasets.pkl` has no schema guard and is now stale. It is loaded only by `main.py --evaluate`
  (which fails at the checkpoint check first) and by `train()` when it is called without fresh
  data.
- `research/benchmark_results.json` was not modified. Its contents describe the schema-2
  simulator.

## Risks and Phase 2 follow-ups
1. **Output filenames.** `model/train.py` always writes `checkpoints/best_model.pt`, and
   `research_benchmark.py` always writes `research/benchmark_results.json`. Both would break the
   no-overwrite rule on the first retrain or rerun. Add output-name arguments first, e.g.
   `best_model_v3.pt` and `benchmark_results_v4.json`.
2. **Persistence vs `label_mode`.** `research_benchmark.py` computes binary persistence as
   `edge_features[..., 0] > 0.85`. That only matches `label_mode = "offered"` and must change
   if the default switches.
3. **Metric semantics.** `metrics()['avg_utilization']` and `['avg_latency']` now average over
   96 directed links, so they can't be compared with schema-2 numbers.
4. **Node features.** Node features now describe egress queues only, which changes the GNN's
   input semantics. A retrain is required anyway.
5. **Docs and paper.** These still describe "one shared FIFO per physical link":
   `README.md:13`, `explanation.tex:40`, `computer_networks.tex:95`, `MODEL_AUDIT.md:33`. Update
   them together with the re-run results.
6. **Fabric-only metric.** With 67% of drops on host links, report fabric-link drop reduction
   separately, alongside the onset metric still missing from CLAUDE.md.

## Verification performed
- `python -m unittest discover -s tests -v`: 20 tests, OK.
- `python tools/guard_ablation_check.py`: completed, output as above. Every run asserts
  `drop_split_ok`.
- Label-mode comparison: a throwaway script, not committed.
- `routing.comparator.run_comparison` on hotspot seed 999 with 200 ticks and no model:
  - ECMP: 375,712 drops (175,669 host / 200,043 fabric).
  - Reactive: 339,006 drops (197,141 host / 141,865 fabric), 901 reroutes.
  - Conservation and the drop split hold in both runs.

## Diff stat (excluding this report)
```
 CLAUDE.md                     | 18 ++++++--
 config.py                     | 14 ++++---
 dashboard/server.py           | 14 +++----
 main.py                       |  5 +--
 model/gnn_model.py            | 10 ++++-
 model/train.py                |  5 ++-
 research_benchmark.py         |  3 +-
 routing/guarded.py            |  6 ++-
 routing/rerouter.py           | 20 ++++-----
 simulation/network_engine.py  | 40 +++++++++++++++---
 simulation/telemetry.py       | 45 ++++++++++++--------
 tests/test_dashboard.py       | 10 +++--
 tests/test_research.py        | 98 ++++++++++++++++++++++++++++++++++++++++++-
 tools/guard_ablation_check.py | 16 +++++--
 14 files changed, 238 insertions(+), 66 deletions(-)
```
