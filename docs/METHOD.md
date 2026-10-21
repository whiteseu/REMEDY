# Method correspondence

## CKM: a persistent retrieval state

For the K retrieved exemplars, let `a_i = max(cosine_i, 0)`. Normalize affinities to weights `w_i`; when all affinities are zero, use `1/K`.

- `r_c = sum_i w_i * 1[c in y_i]`: similarity-weighted support for final RSF.
- `q_c = sum_i 1[c in y_i] / K`: unweighted vote for disagreement and RSF weights.
- `C_ret = {c: q_c > 0}`.
- `d_vote = mean_{c in C_ret} 4*q_c*(1-q_c)`, with zero for an empty raised set.

`state.py` implements these equations. Routing never substitutes `r` for `q`.

## Planning and candidate admission

`pipeline.py` projects task identifiers, view-availability flags, `r`, `q`, disagreement, retrieved labels/similarities and attached grounding to RCMP. Raw query pixels, clinical context, image paths and reference labels do not enter that projection.

The admissible set is `C_ret` union named differential links in the retrieved grounding, restricted to direct labels. Candidate and unresolved sets are filtered deterministically. The expanded grounding `G+` includes entries for **every admissible label**, including differentials absent from retrieved labels. T6, T7 and ECC receive `G+`. Candidate admission constrains planning, not the scored vocabulary.

## Tools and routing

| Stage | Inputs |
| --- | --- |
| T2 | query image/rendering and planned verification targets |
| T3 | clinical text/measurement record and planned targets |
| T4 | validated T2/T3 records |
| T5 | available T2/T3 records and retrieved neighborhood |
| T6 | candidates, collected structured records and G+ |
| T7 | candidates, collected records including T6, and G+ |
| ECC | plan, all executed structured records, direct vocabulary and G+ |

The raw neighborhood is visible downstream only to RCMP and T5. The image goes only to T2. An unreadable or failed required perception record counts as missing evidence. A usable normal read does not count as missing evidence just because it has no positive findings.

DDTR is the OR of: `d_vote >= tau_vote`, `cross_conflict >= tau_cross` or missing required evidence, and high planner complexity. An open gate admits T6; T7 additionally requires the planner's counterfactual request. A counterfactual request by itself does not open the gate. The base path contains the available perception tools, T4 when both perception records are usable, and T5.

With both usable base views and no repairs/fallbacks, the pruned/full paths make 6/8 model calls. CKM and RSF make no model calls. All stages use one configured frozen backbone with temperature zero.

## ECC and failure behavior

ECC receives the manuscript's five fixed rules in `prompts.py`. The model performs adjudication; code validates its structure, finite bounded strengths/scores, complete label coverage and at least one evidence reference to an executed usable tool behind every direct-label score. These checks enforce contracts, not clinical correctness.

Each model stage allows one repair retry. After failure:

- RCMP uses an equal-priority base plan and disables deep stages, matching the default-plan ablation.
- An evidence tool produces an explicitly unusable empty record.
- ECC falls back to the shared direct score readout Pi, using the produced context and the same complete score-vector contract.
- If Pi also fails, its scores are all `0.5`; this is flagged as degraded. Final RSF still follows the configured method.

No failure is quietly converted to a healthy normal read, an omitted zero score or a retrieval-only ECC score. The result records fallbacks, schema attempts and the score source. Model transport retries are separate from schema repair attempts.

## Leave-one-label-out RSF

For each directly scored coordinate `c`:

```text
d_vote_without_c = mean_{j in C_ret excluding c} 4*q_j*(1-q_j)
                  (zero if that set is empty)
alpha_c         = alpha0 * (1 - d_vote_without_c)
s_final_c       = (1-alpha_c)*s_mod_c + alpha_c*r_c
prediction      = labels whose final score is at least delta
```

The vote on `c` does not affect the weight of its own support. Unraised coordinates are not inserted into the mean. A single raised coordinate gets the full base weight for its own support. `state.py` also provides the BRSET derived-coordinate helper; that derived coordinate must not be included among directly fused scores.

Defaults are K=5, `tau_vote=0.40`, `tau_cross=0.50`, `alpha0=0.30`, and `delta=0.50`. The CXR demonstration uses the 14-label CheXpert vocabulary. For the manuscript's MIMIC-CXR benchmark, macro-AUROC and macro-AUPRC are averaged over all 14 labels, including No Finding and Support Devices, matching the label space in Table 1. This release does not include cohort selection or target-trained baselines.

## Release boundaries

The release has one CXR demonstration entry point and generic image/text core stages. ECG waveform processing/rendering, fundus label mappings, knowledge content, target cohorts and benchmark results are not distributed. Core tests create mathematical toy fixtures in code and use fake model responses. Passing them verifies implementation contracts, not published benchmark performance.
