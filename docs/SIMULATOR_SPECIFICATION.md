# Simulator Specification

This document records the implemented symbolic simulator used by the manuscript so that the paper can cite the GitHub repository while keeping reproducibility details public.

## Scenario Generator

The simulator uses three profile labels with equal probability: anti-UAV, open conflict-event, and humanitarian C2. The profiles differ in source-category labels, claim-type bias, uncertainty shift, and complexity shift. They do not reproduce real operations.

| Variable | Anti-UAV | Open conflict-event | Humanitarian C2 |
| --- | --- | --- | --- |
| Profile probability | Uniform over three profiles | Uniform over three profiles | Uniform over three profiles |
| Sources | anti_uav, dut_anti_uav, antiuav600, mmaud, osm | ucdp_ged, acled, natural_earth | hdx_cod, osm, natural_earth |
| Claim-type bias | timing, risk, resource, rule | risk, timing, intent, constraint | resource, constraint, timing, task |
| Complexity c | min(9, randint(3,8)+1) | min(9, randint(3,8)+1) | randint(3,8) |
| Uncertainty xi | clip(0.24 + U[0,0.55] + 0.12) | clip(0.24 + U[0,0.55] + 0.16) | clip(0.24 + U[0,0.55] + 0.04) |
| Time pressure tau | 0.20 + U[0,0.62] | same | same |
| Claim count | 18 + 6c + randint(0,11) | same | same |
| COA count | randint(3,5) | same | same |
| Claim type | biased type with probability 0.64, otherwise uniform over seven types | same | same |
| Reliability rho | clip(0.34 + U[0,0.60] - 0.08 if claim type is risk) | same | same |
| Conflict-prone flag | Bernoulli(0.45 xi) for resource/timing/risk, Bernoulli(0.24 xi) otherwise | same | same |
| Linked-claim probability | 0.20 + 0.025c per claim | same | same |
| COA feasibility F | clip(0.44 + U[0,0.42] - 0.10 xi) | same | same |
| COA intent I | clip(0.50 + U[0,0.36]) | same | same |
| COA baseline risk R_base | clip(0.20 + U[0,0.54] + 0.11 tau) | same | same |

## Policy Parameters

| Policy | p_e | p_q | p_g | p_r | p_f | p_l |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct-generation proxy | 0.36 | 0.18 | 0.14 | 0.22 | 0.34 | 1.20 |
| Evidence only | 0.78 | 0.22 | 0.20 | 0.30 | 0.42 | 1.12 |
| Evidence + role | 0.78 | 0.30 | 0.36 | 0.78 | 0.74 | 1.02 |
| Evidence + critique | 0.78 | 0.78 | 0.40 | 0.42 | 0.50 | 1.08 |
| Full framework / Ours | 0.86 | 0.82 | 0.90 | 0.86 | 0.86 | 0.88 |
| COA-GPT-inspired proxy | 0.40 | 0.22 | 0.18 | 0.25 | 0.36 | 1.15 |
| RAG-inspired proxy | 0.78 | 0.30 | 0.24 | 0.34 | 0.45 | 1.08 |
| AutoGen-inspired proxy | 0.56 | 0.62 | 0.32 | 0.70 | 0.66 | 1.02 |
| MASMP-inspired proxy | 0.60 | 0.46 | 0.42 | 0.74 | 0.76 | 0.98 |

Parameter meanings:

- p_e: evidence-capture intensity.
- p_q: critique-questioning rate.
- p_g: commander-gate strictness.
- p_r: role-boundedness.
- p_f: stage-aware information-flow control.
- p_l: information-load factor.

## Evidence State Rules

For claim reliability rho_i, policy evidence parameter p_e captures the claim with probability:

```text
P(captured_i = 1) = p_e * (0.65 + 0.35 * rho_i)
```

Support-state priority:

1. Uncaptured claims are marked assumed.
2. Captured conflict-prone claims are marked conflicted with probability 0.65.
3. Otherwise, captured claims with rho_i < theta_e are marked assumed with probability 0.55.
4. Otherwise, captured claims are expired with probability 0.04.
5. Remaining captured claims are verified.

Thus a captured claim that is both conflict-prone and below theta_e is first exposed to the conflict-prone rule; only if that rule does not mark it conflicted can the threshold rule mark it assumed.

Claim confidence is `clip(0.25 + 0.65*rho_i)` if captured and `0.18` if uncaptured.

## Control, Critique, and Selection Rules

For flow-control parameter p_f, role-boundedness p_r, and gate strictness p_g:

```text
C_f = clip(0.25 + 0.55 p_f + 0.18 p_r - 0.08 tau + U[0,0.06])
C_r = clip(0.18 + 0.76 p_r + U[0,0.04])
C_g = clip(0.08 + 0.90 p_g - 0.05 xi)
V_a = max(0, round((1-p_g) * |Y| * (0.8 + tau)))
```

For candidate Y, fragile claims are linked claims that are unverified or conflict-prone:

```text
mu_Q = |fragile(Y)| * 5 p_q * (0.05 + 0.05 xi)
Q = max(0, round(mu_Q + U[0,2]))
S_Q = round(Q * (0.18 + 0.35 xi))
A_Q = round(S_Q * (0.35 + 0.55 p_g))
```

Critique coverage is `clip(Q / max(1, 0.32*|L_Y|))`.
Critique effectiveness is `A_Q / max(1, S_Q)`.
Unresolved-risk rate is `(S_Q - A_Q) / max(1, S_Q)`.

## Main Metrics

Let `cl(x)=clip_[0,1](x)`. Let r_capt, r_ver, and r_unc be the captured, verified, and uncertain ratios among claims linked to Y.

```text
T = cl(0.62 r_capt + 0.38 r_ver)
B = cl(0.45 + 0.25 Q_c + 0.20 r_ver - 0.20 r_unc - 0.06 xi)
C = cl(0.42 C_r + 0.32 C_f + 0.26 C_g - 0.035 V_a)
L = cl(p_l * (0.22 + 0.06 c + 0.12 T + 0.16 Q_c + 0.08 r_unc))
R_p = cl(R_base + 0.16 r_unc - 0.008 A_Q)
GPI = cl(0.22F + 0.18I + 0.20B + 0.20T + 0.20C - 0.10R_p - 0.15 lambda_l L)
U_syn = cl(0.48F + 0.32I + 0.20(1-R_base) - 0.045xi - 0.035tau - 0.015c/9
           + 0.045F(1-xi) + 0.035I(1-tau))
```

Each policy selects the candidate with maximum GPI. Oracle regret is computed against the candidate with maximum U_syn in the same generated candidate set.

Non-verified linked-claim ratio is the fraction of linked claims in the selected COA whose state is not verified. It is an uncertain/assumed/conflicted/expired linked-claim ratio, not an assertion that mandatory decision-critical claims are accepted without disclosure.

Low-reliability verified-claim proportion is the fraction of all linked claims that are verified despite reliability below 0.55. Its denominator is all linked claims, not only low-reliability claims.

Severe-issue acceptance ratio is the ratio of accepted severe issues to all detected severe issues.

## Parameter Calibration and Seeds

The released configuration uses separate deterministic streams for calibration, validation, and test generation:

- calibration seed: `20260710`
- validation seed: `20261710`
- test seed: `20262710`

All nine L9 parameter rows are evaluated on the same 2,700 calibration scenarios. The selected default is a calibration-informed conservative default, not the unconstrained range-optimal setting. The pre-test constraints are `theta_e >= 0.55` and `p_q <= 0.82`.

The sensitivity diagnostic uses 20 stochastic replications on a fixed 500-scenario subset. Within each replication, all six settings share common random numbers and the same stochastic stream. The factorial diagnostic uses 10 stochastic replications on a fixed 1,000-scenario subset. Within each replication, all 32 combinations share common random numbers and the same stochastic stream. These diagnostics isolate parameter and mechanism changes from mechanism-level stochastic variation; they should not be read as independent scenario resampling. The exported factorial effects report seed-level means, seed-level standard deviations, and observed seed-level ranges.

## Alternative Rule-Audit Evaluator

The alternative audit evaluator is not used for COA selection. It is a weighting-robustness diagnostic, not an independent external validator.

Let W be low-reliability verified-claim proportion and R_u unresolved-risk rate:

```text
E_p = cl(1 - W - 0.50 r_unc)
A_r = cl(0.52 + 0.18 C_g + 0.16 Q_eff + 0.10 E_p - 0.08 V_a)
A_f = cl(F - 0.07c/9 - 0.06tau)
A_t = cl(I + 0.08C_f - 0.12tau - 0.08R_u)
A_risk = cl(R_base + 0.12r_unc + 0.08xi + 0.05W)
A_alt = cl(0.27A_r + 0.25A_f + 0.23A_t + 0.25(1-A_risk))
```

The released diagnostic file `outputs/experiment_alternative_audit_diagnostics.csv` reports Pearson and Spearman correlation between GPI and A_alt, scenario-level top-1 agreement, and full-framework top-1 rates under both evaluators.

## Exported Files

The main script exports raw and summary CSV files under `outputs/`. The most important files are:

- `experiment_policy_results.csv`
- `scenario_records.csv`
- `scenario_claims.csv`
- `candidate_coas.csv`
- `experiment_comparison_summary.csv`
- `experiment_public_method_results.csv`
- `experiment_public_method_summary.csv`
- `experiment_full_metric_summary.csv`
- `experiment_sensitivity_summary.csv`
- `experiment_sensitivity_seed_summary.csv`
- `experiment_sensitivity_paired_deltas.csv`
- `experiment_seed_batch_summary.csv`
- `experiment_seed_batch_tests.csv`
- `experiment_sesoi_band_checks.csv`
- `experiment_factorial_design.csv`
- `experiment_factorial_effects.csv`
- `l9_calibration_results.csv`
- `l9_range_analysis.csv`
- `l9_validation_results.csv`
- `experiment_alternative_audit_diagnostics.csv`
- `experiment_scalability.csv`
- `coa_timeline_overlay_case.csv`
- `coa_timeline_baseline_comparison.csv`

## Reported Runtime Environment

The scalability table in the manuscript was generated on Microsoft Windows 10 Pro 10.0.19045, Intel Core i7-9750H CPU at 2.60 GHz with 6 cores and 12 logical processors, approximately 32 GB RAM, and Python 3.9.7. The scalability timer uses a single Python process with no multiprocessing and measures scenario generation plus symbolic policy evaluation in `run_experiment`. It excludes CSV writing, figure generation, L9 calibration, COA-timeline rendering, LLM inference, vector retrieval, network calls, interface latency, and human review.
