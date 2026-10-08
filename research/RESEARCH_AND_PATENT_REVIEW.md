# NeuralFabric - Research and Preliminary Patent Review

**Prepared: October 2026**

This is a technical prior-art screen, not a legal opinion, freedom-to-operate analysis, or patentability determination. A missing result is not evidence of novelty. Claim construction, family prosecution history, Indian national-phase status, ownership, and inventorship require a patent professional.

## 1. Technical position of the project

The broad concept "predict congestion from telemetry and reroute traffic" is already represented in prior research and patent literature. The strongest defensible contribution of the supplied project is narrower:

> an empirical evaluation of a forecast-driven rerouting policy whose route changes are admitted only after explicit risk, capacity, freshness, cooldown, path-stretch, and per-tick change-budget checks.

Even that should be presented as an engineering hypothesis because the current evaluation is synthetic and small.

## 2. Important research references

| Work | Relevant overlap | Correct interpretation for NeuralFabric |
|---|---|---|
| ECMP / RFC 2992 | Multipath baseline | Use the correct title: **Analysis of an Equal-Cost Multi-Path Algorithm**. |
| CONGA | Telemetry-driven congestion-aware load balancing | Do not claim reactive schemes are universally too slow to respond; CONGA is specifically designed for datacenter reaction and should be compared by implementation, not assertion. |
| RouteNet / RouteNet-Fermi | GNNs for network modelling and performance prediction | GNN-based network prediction is established. A GCN-LSTM architecture by itself is not a novelty claim. |
| HPCC / PINT | In-band telemetry for congestion control and measurement | INT-style telemetry plus ML is an established direction. |
| GraphCC (Computer Networks, 2025) | GNN + control for datacenter congestion | GraphCC uses GNNs with multi-agent reinforcement learning to optimize ECN-based congestion control; it is not the same route-admission controller. |
| NTP-INT (JNCA, 2025) | Traffic prediction + telemetry planning | NTP-INT uses MTGNN-based traffic prediction plus topology pruning and probe-path planning. This is a close conceptual neighbor for prediction-driven telemetry, but it is not the same guarded route-change mechanism. |
| RouteNet-TGNN (Journal of Communications, 2026) | Temporal GNN forecasting of delay/loss | This confirms that temporal graph prediction in dynamic networks is already an active research area. |
| Hopper (2025 preprint) | Congestion-aware path switching for AI/RDMA clusters | AI-cluster routing and switching-time constraints are already being studied; a production claim would require transport/order evaluation. |
| FLB (USENIX ATC 2025) | Fine-grained load balancing with PFC interactions | Rerouting can interact badly with lossless-network queue dependencies. NeuralFabric therefore must not claim PFC safety without modeling PFC. |

### Primary-source links

- RFC 2992: https://www.rfc-editor.org/rfc/rfc2992
- RouteNet: https://arxiv.org/abs/1910.01508
- GraphCC: https://doi.org/10.1016/j.comnet.2024.110981
- GraphCC preprint: https://arxiv.org/abs/2308.04905
- NTP-INT: https://doi.org/10.1016/j.jnca.2025.104265
- RouteNet-TGNN: https://www.jocm.us/show-324-2145-1.html
- Hopper: https://arxiv.org/abs/2506.08132

## 3. Preliminary patent screen

The following records are useful search leads because they contain elements that overlap with the broad architecture.

| Record | Technical content relevant to this project | What is different / unresolved |
|---|---|---|
| US11470005B2 | Predict expected application bitrates, obtain telemetry, detect path congestion, and reroute traffic. | NeuralFabric predicts a per-link congestion label/score and uses a different simulator/controller implementation. This is not a legal novelty distinction. |
| US11895008B1 | Predictive routing using smoothed network time-series and dynamically selected smoothing envelopes. | NeuralFabric uses a fixed temporal window and GCN-LSTM; it does not implement envelope selection. Claim-level overlap still needs professional review. |
| US12028239B2 | Cross-application predictive routing with path-quality prediction, congestion-risk modelling, constrained optimization, and supervisory control. | NeuralFabric is a synthetic datacenter-style path admission experiment and does not implement the same application-aware optimization. This record demonstrates that predictive constrained routing is already claimed in a broad family of network settings. |

Primary records:

- https://patents.google.com/patent/US11470005B2/en
- https://patents.google.com/patent/US11895008B1/en
- https://patents.google.com/patent/US12028239B2/en

## 4. India-specific review notes

The Indian Patent Office currently lists the **Guidelines for Examination of Computer Related Inventions (CRIs) - 2025**, published 29 April 2026, and separate **Guidelines for the Use of Artificial Intelligence in Patent Examination Procedures**, published 7 August 2026.

Official source page:

https://ipindia.gov.in/resource/patents-resources-guidelines

Official CRI 2025 PDF:

https://ipindia.gov.in/frontend/pdf/patents/guidelines/GUIDELINES%20FOR%20EXAMINATION%20OF%20COMPUTER%20RELATED%20INVENTIONS%20%28CRIs%29%20-%202025.pdf

These guidelines should be read together with the Patents Act and current case law. The project should not state that an algorithm or software implementation is "patentable" merely because it produces a technical result in a simulator.

## 5. Inventorship / disclosure checklist

Before public disclosure or filing, record:

1. Which person designed each algorithmic component.
2. Which person wrote or substantially modified each code module.
3. Which person designed experiments, datasets, or controller safeguards.
4. What was independently developed versus copied, adapted, or inspired by prior work.
5. Dates of conception, implementation, testing, and internal review.
6. Whether the university or another organization has an IP policy that affects ownership.

A patent agent should then prepare a claim chart against the closest references and verify the relevant Indian and international families.

## 6. Project-specific novelty risk

The highest novelty risk is the combination at a high level: GNN/temporal prediction + telemetry + rerouting + constrained optimization/capacity checks. Each ingredient appears in related research or patents. The project therefore needs a much more specific technical distinction to support any future patent strategy.

A promising direction for future research is not to claim novelty now, but to make the guard formally measurable: define a complete admission invariant, prove what it guarantees under the simulator's traffic model, and evaluate ablations showing that the guard—not simply the predictor—causes the stability improvement. Then compare against strong implemented baselines under the same packet-level semantics.

## 7. Bottom line

**Do not submit the phrase "novel patented solution" or a similar claim.** Submit the project as a reproducible experimental study unless and until the research and IP review provide evidence for a narrower technical invention.
