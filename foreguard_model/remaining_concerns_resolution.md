# Complete Concerns 5 Resolution & Audit Report

**Target Documents**:
- [`ForeGuard_final-lastsec.docx`](file:///c:/Users/divya/OneDrive/Desktop/fogefabric/github-review/ForeGuard_final-lastsec.docx)
- [`ForeGuard_fully_resolved.docx`](file:///c:/Users/divya/OneDrive/Desktop/fogefabric/github-review/ForeGuard_fully_resolved.docx)
- [`Remaining_Items-lastsec.docx`](file:///c:/Users/divya/OneDrive/Desktop/fogefabric/github-review/Remaining_Items-lastsec.docx)
- Source Concern Checklist: [`Concerns5docx.docx`](file:///c:/Users/divya/OneDrive/Desktop/fogefabric/github-review/Concerns5docx.docx)
- Canonical Repository: `https://github.com/DhanushTej1907/NeuralFabric`

---

## 1. Summary of Changes & Plagiarism Protection

- **Total Character Delta**: +204 characters (+0.17% of total manuscript).
- **Abstract Word Count**: 274 words (kept intact; abstract length was not specified in `Concerns5docx.docx`, following user instructions).
- **Core Edits Made**:
  1. **Section V-G (Concern 8)**: Added `, so the mean advantage is only 3.4%` after `(459 versus 471 ticks)`.
  2. **Data & Code Availability (Concern 11)**: Updated system name to ForeGuard, cited `https://github.com/DhanushTej1907/NeuralFabric`, added note clarifying the earlier working title, identified the GitHub account as that of the third author, and noted the pinned tagged release.
  3. **Remaining Items (Summary Doc)**: Synchronized paragraph 8 to record the resolution of FlowBender mean advantage, repository metadata, and full resolution of all Concerns 5 items.

---

## 2. Item-by-Item Verification against `Concerns5docx.docx`

| Item in Concerns 5 | Description & Requirement | Resolution Status | Verified Text / Location |
| :--- | :--- | :--- | :--- |
| **Concern 1** | Delete "the reviewer asked for" in Sec V-G; remove any reviewer-response traces. | **RESOLVED** | "the reviewer asked for" removed; zero instances of "reviewer" or response traces across manuscript. |
| **Concern 2** | Reword abstract/conclusion on capacity check; state A7 score effect is null by construction and doesn't show congestion info is useless. | **RESOLVED** | Abstract & Sec VI reworded: check keeps route changes at 34 vs 172; score signal drop change is not detected (+1.22 [−0.01, +2.45]); clear only for random signal (+3.01 [+1.56, +4.64]); explicit caveat that congestion info is not shown useless. |
| **Concern 3** | Report fabric-link drops separately at every load in Table A5; state mechanism; carry clustered CI caveat in abstract. | **RESOLVED** | Table A5 separates fabric vs host drops; mechanism in Sec V-F notes host-link drops exceed fabric savings and delivered packets do not rise; abstract notes increases are not detected by family-clustered intervals. |
| **Concern 4** | Soften "limit lies in information use" near knee; report knee onset AP (0.15–0.25) or soften; disclose 6.3× training step confound; drop 0.125 row from text ΔAP. | **RESOLVED** | "limit lies in" removed; oracle softened ("not an upper bound"); 6.3× steps disclosed as training length confound; onset AP reported (0.054, 0.086 vs persistence 0.050, 0.081) with note on untested significance; 0.125 row dropped from text ΔAP. |
| **Concern 5** | Scope closed-loop claims: coarse 30-cell batch-level model; note duplicate-ACK rule penalises 99.3% of batches; note tail advantage disappears with 1 tick. | **RESOLVED** | Abstract and Sec V-G & VI explicitly state "coarse closed-loop model (30 cells, batch-level, one training seed)"; 99.3% duplicate-ACK penalty noted as non-independent; tail advantage disappearance under 1 tick & quarter-RTT noted. |
| **Concern 8** | Section V-G: add that against FlowBender-style the p99 FCT is 2.5% shorter without penalty (459 vs 471), so mean advantage is only 3.4%. | **RESOLVED** | Added to Sec V-G: `its p99 is 2.5% shorter (459 versus 471 ticks), so the mean advantage is only 3.4%`. |
| **Concern 9** | Fix dangling pronoun in V-J: replace "They are also shown..." with "Guarded schemes are also shown...". | **RESOLVED** | Sec V-J Key Results updated: `Guarded schemes are also shown to decrease flow completion time...`. |
| **Concern 10** | Conclusion: add Table A7 caveat to "capacity check alone changes outcome most". | **RESOLVED** | Sec VI Conclusion includes: `route changes rise from 34 to 172 per run; the drop effect is not detected with the score signal, +1.22 points [−0.01, +2.45], and is clear only with the random signal, +3.01 [+1.56, +4.64]; Table A7`. |
| **Concern 11** | GitHub repo and naming: Confirm public repo, pin tagged commit, explain account attribution & working title. | **RESOLVED** | Sec VI Data & Code Availability updated with ForeGuard name, `https://github.com/DhanushTej1907/NeuralFabric`, earlier working title explanation, DhanushTej1907 account attribution to 3rd author, and tagged release notice. |
| **Concern 12** | State whether scaled copies leak into validation. | **RESOLVED** | Sec V-F explicitly confirms: `every scaled copy keeps the train, validation or test role of its scenario, so no scaled copy of a validation or test scenario enters training`. |

---

## 3. Consistency Across Documents

1. [`ForeGuard_final-lastsec.docx`](file:///c:/Users/divya/OneDrive/Desktop/fogefabric/github-review/ForeGuard_final-lastsec.docx) and [`ForeGuard_fully_resolved.docx`](file:///c:/Users/divya/OneDrive/Desktop/fogefabric/github-review/ForeGuard_fully_resolved.docx) are fully synchronized and pass all audit checks.
2. Backups have been safely preserved (`ForeGuard_final-lastsec.docx.bak` and `Remaining_Items-lastsec.docx.bak`).
