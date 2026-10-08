# NeuralFabric - Invention Disclosure Worksheet

**Status: internal technical worksheet; not a patent application**

## 1. Working title

**Capacity-Guarded Predictive Routing for Burst-Tolerant Datacenter-Style Networks**

## 2. Problem addressed

Forecast-driven rerouting can overreact when several flows respond to the same predicted bottleneck. Independent route decisions may consume the same alternative capacity and shift congestion instead of removing it.

## 3. Implemented mechanism

The current prototype combines:

1. Per-link telemetry features collected over a 10-tick history.
2. A two-layer GCN + one-layer LSTM predicting a binary congestion score 5 ticks ahead.
3. A route-risk function based on the maximum predicted edge risk on a candidate path.
4. A conservative capacity test on newly added links.
5. Sequential reservation after every accepted move.
6. A minimum predicted-risk improvement requirement.
7. A maximum hop-stretch rule.
8. Prediction freshness, per-flow cooldown, and per-tick route-change limits.

## 4. Current guard invariant in the simulator

For each newly added physical link $e$:

`D(e) + moved_flow_rate + queued_packets(e) <= 0.90 * capacity(e)`

where `D(e)` is initialized from the maximum of active-flow demand and observed arrivals. Accepted moves update the reservation state before the next move.

This is a simulator-specific admission invariant. It is **not** a physical bandwidth guarantee and does not account for transport dynamics, switch-programming delay, full-duplex semantics, or PFC/ECN behavior.

## 5. Experimental evidence

Recorded pilot benchmark:

- 30 data scenarios.
- 21/4/5 scenario-group train/validation/test split.
- Six synthetic traffic families.
- Three paired routing seeds per family.
- 90 routing runs total.
- 5.94% mean drop reduction versus ECMP over 18 matched trials.
- 15 wins, 3 ties, 0 losses versus ECMP.
- 86.4% reduction in stride workload drops.
- 97.6% fewer route mutations than reactive rerouting.

## 6. Evidence that limits the claim

- Persistence has higher F1 than GCN-LSTM at the fixed 0.5 cutoff.
- Mixed-workload drop reduction is only 0.49%.
- The GNN does not consistently beat persistence or reactive routing.
- The benchmark has only three seeds per family.
- All experiments use one k=4 topology.
- The simulator uses a 1 ms tick and 500 packets/tick.
- No PFC, ECN, RoCEv2 transport, RDMA ordering, NCCL, or hardware switch programming is modeled.
- Inference p99 is 6.185--13.516 ms on the recorded host and some tails exceed the 5 ms forecast horizon.

## 7. Prior-art questions to answer before any filing

A professional search should ask whether the exact combination of:

- temporal graph congestion prediction;
- path-risk scoring;
- explicit alternate-link capacity reservation;
- sequential multi-flow reservation;
- route-change budget;
- prediction freshness and cooldown; and
- minimum predicted improvement

is disclosed or rendered obvious by existing routing, congestion-control, telemetry, load-balancing, and predictive-routing literature or patents.

Relevant search leads are listed in `RESEARCH_AND_PATENT_REVIEW.md`.

## 8. Inventorship record

Do not infer inventorship from the author list alone. Record which human contributor conceived and substantially developed each claimed technical feature, with dates and supporting artifacts.

## 9. Filing decision

**Current status: research prototype; no patentability conclusion.**

Before public disclosure, route any proposed filing through the appropriate VIT IP/technology-transfer channel and a registered Indian patent agent.
