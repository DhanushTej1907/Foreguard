# NeuralFabric – round 3 change log (fixes to the reworded version)

Only these edits were made; each changed piece of text is blue in the paper. Deleted text (repeated sentences, the pasted fragment) leaves no blue mark.

| # | Paragraph | Original | Revised |
|---|---|---|---|
| 1 | Abstract—In datacenter fabrics | a graph convolutional network combined with a long short-term memory network (GCN-LSTM) | a graph convolutional LSTM network (GCN-LSTM) |
| 2 | Abstract—In datacenter fabrics | The same capacity-guarded controller, which admits a route change only if it lowers path risk and passes stretch, cooldown, budget and sequential capacity checks, is driven | The same capacity-guarded controller is driven |
| 3 | Abstract—In datacenter fabrics | Leave-one-rule-out ablations change the outcome materially only when the capacity check is removed, but adding the rules one at a time shows that they are largely redundant with one another rather than unimportant, and together they give up | Ablations show that the admission rules are largely redundant with one another, and together they give up |
| 4 | The experiments were performed on a note | (with no exception of the ECE column) | (except its ECE column) |
| 5 | The experiments were performed on a note | “Score” and “No Rules” rows from Tables V and IX | the “Score, no rules” rows from Tables V and IX |
| 6 | The Table III below gives | Each batch is made up of 32 samples, with gradient clipping being set at 1.0 so that it can avoid too large gradient values. Training | Training |
| 7 | And we got the 95% confidence | And we got the 95% confidence Interval(CI) which is obtain from 2000 paired bootstrap sample we calculate the calibration from expected calibration error (ECE) , in this 10 equal width probability bins is used | The 95% confidence intervals (CIs) are obtained from 2000 paired bootstrap resamples, and calibration is measured with the expected calibration error (ECE) over ten equal-width probability bins, |
| 8 | where nb is | is number of link window in bin | is the number of link-windows in bin |
| 9 | where nb is | represents the averahe predicted | represent the average predicted |
| 10 | where nb is | Not only depend on packet count , two flow level metrics also used. | Besides packet counts, two flow-level metrics are used: packet reordering (a delivered batch counts as reordered if a later-injected batch of the same flow was already delivered) and, in the closed-loop variant of Section IV-D, the flow completion time (FCT) and its slowdown relative to an idle network. |
| 11 | 1) Rule ablation and sweep parameters: | is guided by a probabilistic risk signal which comprises a static risk and a five-tick oracle look-ahead | is also guided by a random risk signal, a static random risk and a five-tick oracle look-ahead |
| 12 | Table IV and Fig. 2 show the forecaster | the AP for the logitic regression | the AP for the logistic regression |
| 13 | Table IV and Fig. 2 show the forecaster | (vs. 0.052).Calibration has little effect on prediction; the fitted temperature is between 0.79 and 0.98. The maximum change in Brier gain is 0.0016. The effect | (vs. 0.052), so most of the gain over persistence is already available from a linear model. The effect |
| 14 | Table IV and Fig. 2 show the forecaster | However, due to temperature scaling the outputs are still poorly calibrated (Fig. 3). In spite of temperature scaling, because | However, the outputs are still poorly calibrated (Fig. 3): because |
| 15 | To check whether the models learn conges | patterns inThough the five-tick delay does not have statistical significance, it indicates future possible scenarios where forecasting will become an advantagestead of  the traffic generator | patterns instead of the traffic generator |
| 16 | To check whether the models learn conges | In after the every rotation, | In every rotation, |
| 17 | Without a forecaster. | (95% CI [-0.041, 0.068] and 0.058 drops, respectively) | (90% CI [−0.28%, +0.47%] of ECMP drops on the benchmark seeds, Table VI) |
| 18 | Without a forecaster. | maximum edge of 0.10% of ECMP decreases above the current score (95% one-tailed confidence interval) | maximum edge of 0.10% of ECMP drops over the current score (one-sided 95% bound) |
| 19 | Without a forecaster. | current congestion score From the data | current congestion score. From the data |
| 20 | Without a forecaster. | compared to the guarded GNN by an average range of 0.3 to 2.0 family ECMP drops. | compared to the guarded GNN in the four families where they differ, by 0.3 to 2.0 points of the family’s ECMP drops. |
| 21 | Fig. 3 breaks down the drops | Fig. 3 breaks down | Fig. 4 breaks down |
| 22 | Fig. 4 breaks down the drops | for which the guarded criterion removes | for which guarded score removes |
| 23 | Fig. 4 breaks down the drops | then the guarded criterion filters out | then guarded score filters out |
| 24 | Table V also shows the cost | and and display the least amount of delivery time among all methods. (mean: | and display the lowest delivery delay among all methods (mean: |
| 25 | Fig. 6 evaluates the effect of disabling | results in a meaningful reduction in drop reduction: the latter decreases by | results in a meaningful change: the drop reduction shrinks by |
| 26 | Fig. 6 evaluates the effect of disabling | Disabling the minimum improvement rule decreases drops only slightly, | Disabling the minimum improvement rule lowers total drops only slightly further, |
| 27 | Fig. 6 evaluates the effect of disabling | the other two rules overlap rather than implying that they are less important | the other rules overlap rather than implying that they are less important. |
| 28 | Fig. 6. Leave-one-rule-out ablation | (c) Median delivery delay in percentile; | (c) Mean 99th-percentile delivery delay; |
| 29 | Fig. 6. Leave-one-rule-out ablation | uses threshold value 0.7 for validation | replaces the validation threshold by 0.7 |
| 30 | The absence of rules makes visible | shows how much each rule reduces the total reduction in route changes. | shows how much each rule reduces the route changes. |
| 31 | The absence of rules makes visible | the capacity check rule alone continues to reduce the route changes by 4.64% with 134 route changes, | the capacity check rule alone keeps a drop reduction of 4.64% with 134 route changes, |
| 32 | The absence of rules makes visible | and minimum improvement condition brings a 1.61% reduction in route change while | and the minimum improvement condition alone keeps only a 1.61% drop reduction while |
| 33 | TABLE VIIIGuarded score sensitivity | modification budget M | change budget M |
| 34 | TABLE VIIIGuarded score sensitivity | The modifications are compared to ECMP. | Changes are relative to ECMP. |
| 35 | Control signals. | the static random risk per run (6 per run on average) shows no effect. | the static random risk per run moves almost no flows (6 route changes per run) and shows no effect. |
| 36 | Control signals. | representing a drop in the current score by 0.71 and 0.60, respectively (95% confidence interval for each one of these scores lies between [−0.24,+1.72]). | i.e. 0.71 and 0.60 points less than the current score (95% CIs [−0.24, +1.72] and [−0.42, +1.79]). |
| 37 | Reordering. Route change may reorder | flows incur Reordering (score | flows incur reordering (score |
| 38 | Closed-loop flow transport and flow comp | takes a mean FCT of 126.6 ticks compared to ECMP and executes 6,304 route changes in every run.. | takes a mean FCT of 126.6 ticks, longer than ECMP, and executes 6,304 route changes in every run. |
| 39 | Closed-loop flow transport and flow comp | The LLM-learned family is not impacted. | the LLM-training family is not impacted. |
| 40 | Closed-loop flow transport and flow comp | are statistically identical on average | are indistinguishable at the mean |
| 41 | TABLE IXClosed-loop | executed by using the seed for training. | executed by using the first training seed. |
| 42 | TABLE IXClosed-loop | score-based control with admission rules off. | score-based control with admission rules off. FCT and slowdown are over completed flows; censored values are given in the text. |
| 43 | To see whether coarser-grained forecasti | (taking 4.1 ms), Routing result stays preserved; however, in this coarseness, GCN-LSTM inference still stays in line with the simulated clock. Still, complete duration of the controller tick (2.7–3.4 ms on average, 44–51 ms within the family) is out of reach. | (taking 4.1 ms), the routing result stays preserved, and in this coarseness the GCN-LSTM inference stays in line with the simulated clock. Still, the complete duration of the controller tick (2.7–3.4 ms at the median, 44–51 ms within the mixed family) is out of reach. |
| 44 | Load-balancing solutions relying on swit | (FlowBender, Hermes) These functionalities could not be supported by the simulation because | (FlowBender, Hermes) were not implemented in the simulation because |
| 45 | Load-balancing solutions relying on swit | however, other emulations like | however, emulations like |
| 46 | Load-balancing solutions relying on swit | are feasible based on the existing state. | are feasible based on the existing state but were not evaluated. |
| 47 | Load-balancing solutions relying on swit | functioning process.. | functioning process. |
| 48 | NeuralFabric is distinct from reactive a | Our results show that, Under the conditions | Our results show that, under the conditions |
| 49 | NeuralFabric is distinct from reactive a | it is unknown whether CONGA, FlowBender, and Hermes suffer from the same problem. | whether the same holds for CONGA, FlowBender, and Hermes was not tested. |
| 50 | Key results. | are proven to predict | were shown to predict |
| 51 | Interpretation. | in case of adoption of the capacity signal, | in case of adoption of the capacity check, |
| 52 | Interpretation. | without admission signals, | without admission rules, |
| 53 | Interpretation. | becomes 5 drops worse than | becomes 4.9 percentage points worse than |
| 54 | Implications. | does not indicate any benefit of forecasting until delays become equal to five ticks, when the forecaster did worse by 0.72 points. | does not indicate any benefit of forecasting at delays of up to five ticks; with five ticks of actuation delay, the forecaster did worse by 0.72 points. |
| 55 | NeuralFabric examines forecast-driven | a telemetry driven congestion forecaster and model free signals drive the same capacity guarded controller in a full duplex Fat Tree simulation. | a telemetry-driven congestion forecaster and model-free signals drive the same capacity-guarded controller in a full-duplex Fat-Tree simulation. |
| 56 | Future work will test larger topologies | matched to measured control-loop delays | matched to measured control-loop delays, |
| 57 | Future work | (paragraph break) | (merged into one sentence) |
| 58 | Without a forecaster. | 0.18% of ECMP drops, This value is | 0.18% of ECMP drops; this value is |
| 59 | To see whether coarser-grained forecasti | Thus, predicting the future every five ticks, | Thus, when predicting the future every five ticks, |
| 60 | Interpretation. | and delay; The removal of | and delay; the removal of |
| 61 | Interpretation. | lacks of delay, The contribution | lacks of delay, the contribution |
| 62 | The experiments were performed on a note | perfectly reproduced stored values.. | perfectly reproduced stored values. |
| 63 | The Table III below gives | variants of the trained models.. | variants of the trained models. |
| 64 | TABLE IXClosed-loop | RE_TX as percentage | Retx. as percentage |
| 65 | TABLE IXClosed-loop | rerouting (REROUTES) | Reroutes |
