from __future__ import annotations

import csv as csv_lib
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "calibration_config.json"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
REPORT = ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
REPORT.mkdir(parents=True, exist_ok=True)

seed = 20260710


def read_config():
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {
        "test_seed": 20262710,
        "test_scenarios": 10000,
        "public_method_seed": 20262841,
        "ablation_seed": 20262787,
        "sensitivity_seed_base": 20262910,
        "sensitivity_seed_count": 20,
        "sensitivity_scenarios_per_seed": 500,
        "seed_batch_seed_base": 20263710,
        "factorial_seed_base": 20267710,
        "factorial_seed_count": 10,
        "factorial_scenarios_per_seed": 1000,
        "selected_parameters": {"theta_e": 0.55, "p_q": 0.82, "p_g": 0.90, "lambda_l": 0.10},
    }


CONFIG = read_config()
SELECTED = CONFIG.get("selected_parameters", {})

NATURE = {
    "blue": "#3E6FA3",
    "teal": "#5B8E7D",
    "purple": "#8A7FB9",
    "ochre": "#C58B2B",
    "rust": "#B86B3D",
    "slate": "#6F7C8C",
    "green": "#4C8C63",
    "evidence": "#56B4C8",
    "moderate": "#D9A441",
    "high": "#C75146",
    "ink": "#26323F",
    "grid": "#D7DEE8",
}


def set_nature_style(font_size: float = 7.5):
    plt.rcParams.update(
        {
            "font.size": font_size,
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def nature_grid(ax):
    ax.grid(axis="y", color=NATURE["grid"], linewidth=0.45, alpha=0.55)
    ax.set_axisbelow(True)


def rand() -> float:
    global seed
    seed = (1664525 * seed + 1013904223) & 0xFFFFFFFF
    return seed / 0x100000000


def choice(items):
    return items[math.floor(rand() * len(items))]


def randint(a: int, b: int) -> int:
    return math.floor(rand() * (b - a + 1)) + a


def js_round(x: float) -> int:
    return math.floor(x + 0.5)


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


PROFILES = {
    "anti_uav": {
        "label": "Anti-UAV",
        "sources": ["anti_uav", "dut_anti_uav", "antiuav600", "mmaud", "osm"],
        "claim_bias": ["timing", "risk", "resource", "rule"],
        "uncertainty_shift": 0.12,
        "complexity_shift": 1,
    },
    "conflict_event": {
        "label": "Open event",
        "sources": ["ucdp_ged", "acled", "natural_earth"],
        "claim_bias": ["risk", "timing", "intent", "constraint"],
        "uncertainty_shift": 0.16,
        "complexity_shift": 1,
    },
    "humanitarian_c2": {
        "label": "Humanitarian C2",
        "sources": ["hdx_cod", "osm", "natural_earth"],
        "claim_bias": ["resource", "constraint", "timing", "task"],
        "uncertainty_shift": 0.04,
        "complexity_shift": 0,
    },
}

CLAIM_TYPES = ["task", "constraint", "resource", "timing", "risk", "rule", "intent"]

POLICIES = {
    "direct_llm": {
        "label": "Direct-generation proxy",
        "evidence_capture": 0.36,
        "critique_rate": 0.18,
        "gate_strictness": 0.14,
        "role_boundedness": 0.22,
        "flow_control": 0.34,
        "load_factor": 1.20,
    },
    "evidence_only": {
        "label": "Evidence only",
        "evidence_capture": 0.78,
        "critique_rate": 0.22,
        "gate_strictness": 0.20,
        "role_boundedness": 0.30,
        "flow_control": 0.42,
        "load_factor": 1.12,
    },
    "evidence_role": {
        "label": "Evidence + role",
        "evidence_capture": 0.78,
        "critique_rate": 0.30,
        "gate_strictness": 0.36,
        "role_boundedness": 0.78,
        "flow_control": 0.74,
        "load_factor": 1.02,
    },
    "evidence_critique": {
        "label": "Evidence + critique",
        "evidence_capture": 0.78,
        "critique_rate": 0.78,
        "gate_strictness": 0.40,
        "role_boundedness": 0.42,
        "flow_control": 0.50,
        "load_factor": 1.08,
    },
    "full_framework": {
        "label": "Full framework",
        "evidence_capture": 0.86,
        "critique_rate": SELECTED.get("p_q", 0.82),
        "gate_strictness": SELECTED.get("p_g", 0.90),
        "role_boundedness": 0.86,
        "flow_control": 0.86,
        "load_factor": 0.88,
    },
}

PUBLIC_METHOD_BASELINES = {
    "coa_gpt_style": {
        "label": "COA-GPT-inspired proxy",
        "reference": "COA-GPT",
        "evidence_capture": 0.40,
        "critique_rate": 0.22,
        "gate_strictness": 0.18,
        "role_boundedness": 0.25,
        "flow_control": 0.36,
        "load_factor": 1.15,
    },
    "rag_coa_style": {
        "label": "RAG-inspired proxy",
        "reference": "RAG",
        "evidence_capture": 0.78,
        "critique_rate": 0.30,
        "gate_strictness": 0.24,
        "role_boundedness": 0.34,
        "flow_control": 0.45,
        "load_factor": 1.08,
    },
    "autogen_style": {
        "label": "AutoGen-inspired proxy",
        "reference": "AutoGen",
        "evidence_capture": 0.56,
        "critique_rate": 0.62,
        "gate_strictness": 0.32,
        "role_boundedness": 0.70,
        "flow_control": 0.66,
        "load_factor": 1.02,
    },
    "masmp_style": {
        "label": "MASMP-inspired proxy",
        "reference": "MASMP",
        "evidence_capture": 0.60,
        "critique_rate": 0.46,
        "gate_strictness": 0.42,
        "role_boundedness": 0.74,
        "flow_control": 0.76,
        "load_factor": 0.98,
    },
    "ours": {
        **POLICIES["full_framework"],
        "label": "Ours",
        "reference": "This paper",
    },
}

ABLATIONS = {
    "full_framework": dict(POLICIES["full_framework"]),
    "no_evidence_chain": {
        **POLICIES["full_framework"],
        "label": "No evidence chain",
        "evidence_capture": 0.38,
    },
    "no_role_flow": {
        **POLICIES["full_framework"],
        "label": "No role-flow control",
        "role_boundedness": 0.30,
        "flow_control": 0.36,
    },
    "no_adversarial_questioning": {
        **POLICIES["full_framework"],
        "label": "No adversarial questioning",
        "critique_rate": 0.22,
    },
    "no_commander_gate": {
        **POLICIES["full_framework"],
        "label": "No commander gate",
        "gate_strictness": 0.18,
    },
}

PARAMS = {
    "theta_e": SELECTED.get("theta_e", 0.55),
    "lambda_l": SELECTED.get("lambda_l", 0.10),
}

WEAK_EVIDENCE_FLOOR = 0.55


def generate_scenarios(n: int):
    scenarios = []
    profile_keys = list(PROFILES.keys())
    for scenario_id in range(1, n + 1):
        profile_key = choice(profile_keys)
        profile = PROFILES[profile_key]
        complexity = min(9, randint(3, 8) + profile["complexity_shift"])
        uncertainty = clamp(0.24 + rand() * 0.55 + profile["uncertainty_shift"])
        time_pressure = 0.20 + rand() * 0.62
        n_claims = 18 + complexity * 6 + randint(0, 11)
        n_coas = randint(3, 5)
        claims = []
        for i in range(n_claims):
            if rand() < 0.64:
                claim_type = choice(profile["claim_bias"])
            else:
                claim_type = choice(CLAIM_TYPES)
            source = choice(profile["sources"])
            reliability = clamp(0.34 + rand() * 0.60 - (0.08 if claim_type == "risk" else 0.0))
            conflict_multiplier = 0.45 if claim_type in {"resource", "timing", "risk"} else 0.24
            conflict_prone = rand() < uncertainty * conflict_multiplier
            claims.append(
                {
                    "id": f"s{scenario_id}_c{i}",
                    "type": claim_type,
                    "source": source,
                    "reliability": reliability,
                    "conflict_prone": conflict_prone,
                }
            )
        coas = []
        for j in range(n_coas):
            p = 0.20 + complexity * 0.025
            claim_ids = [claim["id"] for claim in claims if rand() < p]
            coas.append(
                {
                    "id": f"s{scenario_id}_y{j}",
                    "claim_ids": claim_ids,
                    "feasibility": clamp(0.44 + rand() * 0.42 - uncertainty * 0.10),
                    "intent": clamp(0.50 + rand() * 0.36),
                    "risk": clamp(0.20 + rand() * 0.54 + time_pressure * 0.11),
                }
            )
        scenarios.append(
            {
                "id": scenario_id,
                "profile_key": profile_key,
                "complexity": complexity,
                "uncertainty": uncertainty,
                "time_pressure": time_pressure,
                "claims": claims,
                "coas": coas,
            }
        )
    return scenarios


def evidence(claims, policy, params):
    out = []
    for claim in claims:
        captured = rand() < policy["evidence_capture"] * (0.65 + 0.35 * claim["reliability"])
        if not captured:
            state = "assumed"
        elif claim["conflict_prone"] and rand() < 0.65:
            state = "conflicted"
        elif claim["reliability"] < params["theta_e"] and rand() < 0.55:
            state = "assumed"
        elif rand() < 0.04:
            state = "expired"
        else:
            state = "verified"
        new_claim = dict(claim)
        new_claim.update(
            {
                "captured": captured,
                "state": state,
                "confidence": clamp(0.25 + 0.65 * claim["reliability"]) if captured else 0.18,
            }
        )
        out.append(new_claim)
    return out


def controls(policy, scenario, params):
    flow = clamp(
        0.25
        + 0.55 * policy["flow_control"]
        + 0.18 * policy["role_boundedness"]
        - scenario["time_pressure"] * 0.08
        + rand() * 0.06
    )
    role = clamp(0.18 + 0.76 * policy["role_boundedness"] + rand() * 0.04)
    gate = clamp(0.08 + 0.90 * policy["gate_strictness"] - scenario["uncertainty"] * 0.05)
    violations = max(
        0,
        js_round((1 - policy["gate_strictness"]) * len(scenario["coas"]) * (0.8 + scenario["time_pressure"])),
    )
    return {
        "flow_compliance": flow,
        "role_compliance": role,
        "gate_closure": gate,
        "authority_violations": violations,
    }


def critique(coa, ev, policy, scenario, params):
    claim_map = {item["id"]: item for item in ev}
    linked = [claim_map[item] for item in coa["claim_ids"] if item in claim_map]
    fragile = [item for item in linked if item["state"] != "verified" or item["conflict_prone"]]
    expected = len(fragile) * 5 * policy["critique_rate"] * (0.05 + scenario["uncertainty"] * 0.05)
    questions = max(0, js_round(expected + rand() * 2))
    severe = js_round(questions * (0.18 + scenario["uncertainty"] * 0.35))
    accepted = js_round(severe * (0.35 + 0.55 * policy["gate_strictness"]))
    unresolved = max(0, severe - accepted)
    coverage = clamp(questions / max(1, len(linked) * 0.32))
    effective = clamp(accepted / max(1, severe))
    return {
        "questions": questions,
        "severe": severe,
        "accepted": accepted,
        "unresolved": unresolved,
        "coverage": coverage,
        "effective": effective,
    }


def simulator_task_utility(scenario, coa):
    base = 0.48 * coa["feasibility"] + 0.32 * coa["intent"] + 0.20 * (1 - coa["risk"])
    stress = 0.045 * scenario["uncertainty"] + 0.035 * scenario["time_pressure"] + 0.015 * (scenario["complexity"] / 9)
    interaction = 0.045 * coa["feasibility"] * (1 - scenario["uncertainty"]) + 0.035 * coa["intent"] * (1 - scenario["time_pressure"])
    return clamp(base + interaction - stress)


def outcome_proxy(scenario, coa):
    return simulator_task_utility(scenario, coa)


def score(coa, ev, crit, ctl, scenario, policy, params):
    claim_map = {item["id"]: item for item in ev}
    linked = [claim_map[item] for item in coa["claim_ids"] if item in claim_map]
    if linked:
        captured = len([item for item in linked if item["captured"]]) / len(linked)
        verified = len([item for item in linked if item["state"] == "verified"]) / len(linked)
        uncertain = len([item for item in linked if item["state"] != "verified"]) / len(linked)
        low_reliability_verified = len(
            [item for item in linked if item["state"] == "verified" and item["reliability"] < WEAK_EVIDENCE_FLOOR]
        ) / len(linked)
        source_diversity = len({item["source"] for item in linked}) / 12
    else:
        captured = verified = uncertain = 0.0
        low_reliability_verified = source_diversity = 0.0
    traceability = clamp(0.62 * captured + 0.38 * verified)
    robustness = clamp(
        0.45 + 0.25 * crit["coverage"] + 0.20 * verified - 0.20 * uncertain - scenario["uncertainty"] * 0.06
    )
    controllability = clamp(
        0.42 * ctl["role_compliance"]
        + 0.32 * ctl["flow_compliance"]
        + 0.26 * ctl["gate_closure"]
        - ctl["authority_violations"] * 0.035
    )
    load = clamp(
        policy["load_factor"]
        * (0.22 + scenario["complexity"] * 0.06 + traceability * 0.12 + crit["coverage"] * 0.16 + uncertain * 0.08)
    )
    risk_penalty = clamp(coa["risk"] + uncertain * 0.16 - crit["accepted"] * 0.008)
    quality = clamp(
        0.22 * coa["feasibility"]
        + 0.18 * coa["intent"]
        + 0.20 * robustness
        + 0.20 * traceability
        + 0.20 * controllability
        - 0.10 * risk_penalty
        - params["lambda_l"] * load * 0.15
    )
    unresolved_risk_rate = clamp(crit["unresolved"] / max(1, crit["severe"]))
    alternative_rule_score = independent_rule_audit(
        coa=coa,
        scenario=scenario,
        ctl=ctl,
        crit=crit,
        uncertain=uncertain,
        low_reliability_verified=low_reliability_verified,
        unresolved_risk_rate=unresolved_risk_rate,
    )
    return {
        "traceability": traceability,
        "robustness": robustness,
        "controllability": controllability,
        "cognitive_load": load,
        "risk_penalty": risk_penalty,
        "quality": quality,
        "uncertain": uncertain,
        "low_reliability_verified": low_reliability_verified,
        "source_diversity": source_diversity,
        "unresolved_risk_rate": unresolved_risk_rate,
        "alternative_rule_score": alternative_rule_score,
    }


def independent_rule_audit(coa, scenario, ctl, crit, uncertain, low_reliability_verified, unresolved_risk_rate):
    """Second evaluator used for robustness auditing, not for COA selection."""
    evidence_precision_proxy = clamp(1.0 - low_reliability_verified - 0.50 * uncertain)
    rule_compliance = clamp(
        0.52
        + 0.18 * ctl["gate_closure"]
        + 0.16 * crit["effective"]
        + 0.10 * evidence_precision_proxy
        - 0.08 * ctl["authority_violations"]
    )
    resource_feasibility = clamp(
        coa["feasibility"] - 0.07 * scenario["complexity"] / 9 - 0.06 * scenario["time_pressure"]
    )
    temporal_consistency = clamp(
        coa["intent"] + 0.08 * ctl["flow_compliance"] - 0.12 * scenario["time_pressure"] - 0.08 * unresolved_risk_rate
    )
    risk_exposure = clamp(
        coa["risk"] + 0.12 * uncertain + 0.08 * scenario["uncertainty"] + 0.05 * low_reliability_verified
    )
    return clamp(
        0.27 * rule_compliance
        + 0.25 * resource_feasibility
        + 0.23 * temporal_consistency
        + 0.25 * (1 - risk_exposure)
    )


def run_policy(scenario, policy_key, policy, params):
    ev = evidence(scenario["claims"], policy, params)
    ctl = controls(policy, scenario, params)
    candidates = []
    for coa in scenario["coas"]:
        crit = critique(coa, ev, policy, scenario, params)
        sc = score(coa, ev, crit, ctl, scenario, policy, params)
        candidates.append({"coa": coa, "crit": crit, "sc": sc})
    selected = sorted(candidates, key=lambda item: item["sc"]["quality"], reverse=True)[0]
    selected_utility = simulator_task_utility(scenario, selected["coa"])
    oracle_ranked = sorted(
        [{"coa_id": coa["id"], "utility": simulator_task_utility(scenario, coa)} for coa in scenario["coas"]],
        key=lambda item: item["utility"],
        reverse=True,
    )
    oracle_utility = oracle_ranked[0]["utility"]
    oracle_top2 = {item["coa_id"] for item in oracle_ranked[:2]}
    mean = lambda fn: sum(fn(item) for item in candidates) / len(candidates)
    linked_selected = selected["sc"]
    return {
        "scenario_id": scenario["id"],
        "profile": scenario["profile_key"],
        "policy": policy_key,
        "policy_label": policy["label"],
        "n_claims": len(scenario["claims"]),
        "n_coas": len(scenario["coas"]),
        "evidence_coverage": len([item for item in ev if item["captured"]]) / len(ev),
        "verified_coverage": len([item for item in ev if item["state"] == "verified"]) / len(ev),
        "critique_coverage": mean(lambda item: item["crit"]["coverage"]),
        "critique_effective": mean(lambda item: item["crit"]["effective"]),
        "severe_issue_acceptance_ratio": mean(lambda item: item["crit"]["accepted"] / max(1, item["crit"]["severe"])),
        "issue_severity": mean(lambda item: item["crit"]["severe"]),
        "flow_compliance": ctl["flow_compliance"],
        "role_compliance": ctl["role_compliance"],
        "gate_closure_rate": ctl["gate_closure"],
        "authority_violations": ctl["authority_violations"],
        "cognitive_load": selected["sc"]["cognitive_load"],
        "low_reliability_verified_claim_proportion": linked_selected["low_reliability_verified"],
        "non_verified_linked_claim_ratio": linked_selected["uncertain"],
        "unresolved_risk_rate": linked_selected["unresolved_risk_rate"],
        "source_diversity": linked_selected["source_diversity"],
        "alternative_rule_score": linked_selected["alternative_rule_score"],
        "selected_gpi": selected["sc"]["quality"],
        "selected_quality": selected["sc"]["quality"],
        "selected_traceability": selected["sc"]["traceability"],
        "selected_robustness": selected["sc"]["robustness"],
        "selected_controllability": selected["sc"]["controllability"],
        "selected_risk_penalty": selected["sc"]["risk_penalty"],
        "selected_synthetic_utility": selected_utility,
        "selected_outcome_proxy": selected_utility,
        "selected_oracle_regret": oracle_utility - selected_utility,
        "oracle_best_hit": 1 if selected["coa"]["id"] == oracle_ranked[0]["coa_id"] else 0,
        "oracle_top2_hit": 1 if selected["coa"]["id"] in oracle_top2 else 0,
    }


def stats(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / max(1, n - 1)
    sd = math.sqrt(variance)
    sorted_values = sorted(values)

    def q(p):
        idx = min(len(sorted_values) - 1, max(0, math.floor(p * (len(sorted_values) - 1))))
        return sorted_values[idx]

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": sd / math.sqrt(n),
        "ci95": 1.96 * sd / math.sqrt(n),
        "q05": q(0.05),
        "q25": q(0.25),
        "median": q(0.5),
        "q75": q(0.75),
        "q95": q(0.95),
    }


def paired_test(rows, policy_a, policy_b, metric):
    by_id = {}
    for row in rows:
        by_id.setdefault(row["scenario_id"], {})[row["policy"]] = row[metric]
    diffs = [item[policy_a] - item[policy_b] for item in by_id.values() if policy_a in item and policy_b in item]
    s = stats(diffs)
    t_stat = s["mean"] / max(1e-12, s["se"])
    p_value = 2 * (1 - normal_cdf(abs(t_stat)))
    return {
        "comparison": f"{policy_a} - {policy_b}",
        "metric": metric,
        "n": s["n"],
        "mean_diff": s["mean"],
        "sd_diff": s["sd"],
        "t_stat": t_stat,
        "p_norm_approx": max(0.0, p_value),
        "cohen_dz": s["mean"] / max(1e-12, s["sd"]),
    }


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def aggregate(rows, group_keys, metrics):
    groups = {}
    for row in rows:
        key = tuple(row[item] for item in group_keys)
        groups.setdefault(key, []).append(row)
    out = []
    for key, vals in groups.items():
        item = {group_keys[i]: key[i] for i in range(len(group_keys))}
        item["n"] = len(vals)
        for metric in metrics:
            s = stats([val[metric] for val in vals])
            item[f"{metric}_mean"] = s["mean"]
            item[f"{metric}_sd"] = s["sd"]
            item[f"{metric}_ci95"] = s["ci95"]
            item[f"{metric}_median"] = s["median"]
            item[f"{metric}_q05"] = s["q05"]
            item[f"{metric}_q95"] = s["q95"]
        out.append(item)
    return out


def write_csv(path: Path, rows, cols):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv_lib.writer(f, quoting=csv_lib.QUOTE_MINIMAL)
        writer.writerow(cols)
        for row in rows:
            values = []
            for col in cols:
                value = row.get(col, "")
                if isinstance(value, float):
                    values.append(f"{value:.6f}")
                else:
                    values.append(value)
            writer.writerow(values)


def scenario_record_tables(scenarios):
    claim_rows = []
    coa_rows = []
    scenario_rows = []
    for scenario in scenarios:
        scenario_rows.append(
            {
                "scenario_id": scenario["id"],
                "profile": scenario["profile_key"],
                "complexity": scenario["complexity"],
                "uncertainty": scenario["uncertainty"],
                "time_pressure": scenario["time_pressure"],
                "n_claims": len(scenario["claims"]),
                "n_coas": len(scenario["coas"]),
            }
        )
        for claim in scenario["claims"]:
            claim_rows.append(
                {
                    "scenario_id": scenario["id"],
                    "claim_id": claim["id"],
                    "claim_type": claim["type"],
                    "source": claim["source"],
                    "reliability": claim["reliability"],
                    "conflict_prone": 1 if claim["conflict_prone"] else 0,
                }
            )
        for coa in scenario["coas"]:
            coa_rows.append(
                {
                    "scenario_id": scenario["id"],
                    "coa_id": coa["id"],
                    "linked_claim_count": len(coa["claim_ids"]),
                    "linked_claim_ids": ";".join(coa["claim_ids"]),
                    "feasibility": coa["feasibility"],
                    "intent": coa["intent"],
                    "risk": coa["risk"],
                }
            )
    return scenario_rows, claim_rows, coa_rows


def make_figures(summary, rows, ablation_summary, profile_summary, scalability):
    set_nature_style(7.5)
    colors = [NATURE["slate"], NATURE["blue"], NATURE["teal"], NATURE["ochre"], NATURE["green"]]
    short_policy_labels = {
        "direct_llm": "Direct",
        "evidence_only": "Evidence",
        "evidence_role": "Evidence+Role",
        "evidence_critique": "Evidence+Critique",
        "full_framework": "Full",
    }

    metrics = [
        "selected_gpi",
        "selected_synthetic_utility",
        "selected_traceability",
        "selected_robustness",
        "selected_controllability",
        "cognitive_load",
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = range(len(metrics))
    width = 0.15
    for i, row in enumerate(summary):
        vals = [row[f"{metric}_mean"] for metric in metrics]
        ax.bar(
            [j + (i - 2) * width for j in x],
            vals,
            width=width,
            label=short_policy_labels.get(row["policy"], row["policy_label"]),
            color=colors[i],
            edgecolor=NATURE["ink"],
            linewidth=0.35,
        )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean score")
    ax.set_xticks(list(x))
    tick_labels = ["GPI", "synthetic\nutility", "traceability", "robustness", "controllability", "load"]
    ax.set_xticklabels(tick_labels)
    ax.legend(ncol=3, frameon=False)
    nature_grid(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig_exp_main_comparison.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    labels = list(POLICIES.keys())
    data = [[row["selected_gpi"] for row in rows if row["policy"] == label] for label in labels]
    box = ax.boxplot(
        data,
        labels=[short_policy_labels[item] for item in labels],
        showfliers=False,
        patch_artist=True,
        medianprops={"color": NATURE["ink"], "linewidth": 1.0},
        whiskerprops={"color": NATURE["ink"], "linewidth": 0.7},
        capprops={"color": NATURE["ink"], "linewidth": 0.7},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
        patch.set_edgecolor(NATURE["ink"])
        patch.set_linewidth(0.6)
    ax.set_ylabel("Governance Process Index")
    ax.tick_params(axis="x", labelrotation=25)
    nature_grid(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig_exp_gpi_boxplot.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    full_gpi = next(row for row in ablation_summary if row["policy"] == "full_framework")["selected_gpi_mean"]
    labels = [row["policy"].replace("_", "\n") for row in ablation_summary]
    drops = [full_gpi - row["selected_gpi_mean"] for row in ablation_summary]
    ax.bar(
        range(len(drops)),
        drops,
        color=[NATURE["green"]] + [NATURE["high"]] * (len(drops) - 1),
        edgecolor=NATURE["ink"],
        linewidth=0.35,
    )
    ax.set_xticks(range(len(drops)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("GPI decrease")
    nature_grid(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig_exp_ablation.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    profiles = list(PROFILES.keys())
    for j, policy in enumerate(["direct_llm", "full_framework"]):
        vals = [
            next(row for row in profile_summary if row["profile"] == profile and row["policy"] == policy)[
                "selected_gpi_mean"
            ]
            for profile in profiles
        ]
        ax.bar(
            [i + (j - 0.5) * 0.32 for i in range(len(profiles))],
            vals,
            width=0.32,
            label=POLICIES[policy]["label"],
            color=[NATURE["slate"], NATURE["green"]][j],
            edgecolor=NATURE["ink"],
            linewidth=0.35,
        )
    ax.set_xticks(range(len(profiles)))
    ax.set_xticklabels([PROFILES[item]["label"] for item in profiles])
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Governance Process Index")
    ax.legend(frameon=False)
    nature_grid(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig_exp_profile_generalization.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.plot(
        [row["scenarios"] for row in scalability],
        [row["runtime_ms"] for row in scalability],
        marker="o",
        color=NATURE["blue"],
        markerfacecolor="white",
        markeredgecolor=NATURE["blue"],
        linewidth=1.4,
    )
    ax.set_xlabel("Scenarios")
    ax.set_ylabel("Runtime (ms)")
    nature_grid(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig_exp_scalability.pdf")
    plt.close(fig)


def make_public_method_figure(public_summary):
    set_nature_style(7.5)
    metrics = [
        "selected_gpi",
        "selected_synthetic_utility",
        "selected_traceability",
        "selected_robustness",
        "selected_controllability",
    ]
    colors = [NATURE["slate"], NATURE["blue"], NATURE["purple"], NATURE["ochre"], NATURE["green"]]
    short_method_labels = {
        "coa_gpt_style": "COA-GPT",
        "rag_coa_style": "RAG",
        "autogen_style": "AutoGen",
        "masmp_style": "MASMP",
        "ours": "Ours",
    }
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = range(len(metrics))
    width = 0.15
    for i, row in enumerate(public_summary):
        vals = [row[f"{metric}_mean"] for metric in metrics]
        ax.bar(
            [j + (i - 2) * width for j in x],
            vals,
            width=width,
            label=short_method_labels.get(row["method"], row["method_label"]),
            color=colors[i],
            edgecolor=NATURE["ink"],
            linewidth=0.35,
        )
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Mean score")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["GPI", "synthetic\nutility", "traceability", "robustness", "controllability"])
    ax.legend(ncol=2, frameon=False)
    nature_grid(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig_exp_public_method_comparison.pdf")
    plt.close(fig)


def run_experiment(n: int = 10000):
    global seed
    seed = CONFIG.get("test_seed", 20262710)
    scenarios = generate_scenarios(n)
    rows = []
    for scenario in scenarios:
        for key, policy in POLICIES.items():
            rows.append(run_policy(scenario, key, policy, PARAMS))
    return scenarios, rows


def run_public_method_comparison(scenarios):
    global seed
    seed = CONFIG.get("public_method_seed", CONFIG.get("test_seed", 20262710) + 131)
    rows = []
    for scenario in scenarios:
        for key, policy in PUBLIC_METHOD_BASELINES.items():
            result = run_policy(scenario, key, policy, PARAMS)
            result["method"] = key
            result["method_label"] = policy["label"]
            result["reference"] = policy["reference"]
            rows.append(result)
    return rows


def run_ablation(scenarios):
    global seed
    seed = CONFIG.get("ablation_seed", CONFIG.get("test_seed", 20262710) + 77)
    rows = []
    for scenario in scenarios:
        for key, policy in ABLATIONS.items():
            rows.append(run_policy(scenario, key, policy, PARAMS))
    return rows


def run_sensitivity(scenarios):
    global seed
    settings = [
        {"name": "low_evidence", "params": {"theta_e": 0.45, "lambda_l": 0.10}, "policy": {}},
        {"name": "selected", "params": {"theta_e": 0.55, "lambda_l": 0.10}, "policy": {}},
        {"name": "high_evidence", "params": {"theta_e": 0.65, "lambda_l": 0.10}, "policy": {}},
        {"name": "low_critique_rate", "params": {"theta_e": 0.55, "lambda_l": 0.10}, "policy": {"critique_rate": 0.62}},
        {"name": "high_critique_rate", "params": {"theta_e": 0.55, "lambda_l": 0.10}, "policy": {"critique_rate": 0.92}},
        {"name": "strict_gate", "params": {"theta_e": 0.55, "lambda_l": 0.10}, "policy": {"gate_strictness": 0.96}},
    ]
    rows = []
    base_seed = CONFIG.get("sensitivity_seed_base", CONFIG.get("test_seed", 20262710) + 300)
    seed_count = CONFIG.get("sensitivity_seed_count", 20)
    scenarios_per_seed = CONFIG.get("sensitivity_scenarios_per_seed", 500)
    scenarios_subset = scenarios[:scenarios_per_seed]
    for seed_idx in range(seed_count):
        common_seed = base_seed + seed_idx * 37
        for setting in settings:
            seed = common_seed
            params = setting["params"]
            policy = {**POLICIES["full_framework"], **setting["policy"]}
            for scenario in scenarios_subset:
                rows.append(
                    {
                        "setting": setting["name"],
                        "sensitivity_seed_index": seed_idx + 1,
                        "sensitivity_seed": common_seed,
                        **run_policy(scenario, "full_framework", policy, params),
                    }
                )
    return rows


def run_seed_batches(num_batches: int = 20, scenarios_per_batch: int = 500):
    global seed
    batch_rows = []
    base_seed = CONFIG.get("seed_batch_seed_base", CONFIG.get("test_seed", 20262710) + 1000)
    for batch in range(num_batches):
        seed = base_seed + batch * 37
        scenarios = generate_scenarios(scenarios_per_batch)
        rows = []
        for scenario in scenarios:
            for key, policy in POLICIES.items():
                rows.append(run_policy(scenario, key, policy, PARAMS))
        summary = aggregate(
            rows,
            ["policy", "policy_label"],
            ["selected_gpi", "selected_synthetic_utility", "selected_oracle_regret", "oracle_best_hit", "oracle_top2_hit"],
        )
        for item in summary:
            batch_rows.append({"batch": batch + 1, "seed": base_seed + batch * 37, **item})
    return batch_rows


def paired_batch_effects(batch_rows, policy_a, policy_b, metric, sesoi=None):
    by_batch = {}
    for row in batch_rows:
        by_batch.setdefault(row["batch"], {})[row["policy"]] = row[f"{metric}_mean"]
    diffs = [item[policy_a] - item[policy_b] for item in by_batch.values() if policy_a in item and policy_b in item]
    s = stats(diffs)
    sorted_diffs = sorted(diffs)
    lo = sorted_diffs[max(0, math.floor(0.025 * (len(sorted_diffs) - 1)))]
    hi = sorted_diffs[min(len(sorted_diffs) - 1, math.floor(0.975 * (len(sorted_diffs) - 1)))]
    out = {
        "comparison": f"{policy_a} - {policy_b}",
        "metric": metric,
        "seed_batches": len(diffs),
        "mean_diff": s["mean"],
        "sd_diff": s["sd"],
        "empirical_p025": lo,
        "empirical_p975": hi,
        "sesoi": "" if sesoi is None else sesoi,
        "within_equivalence_margin": "",
    }
    if sesoi is not None:
        out["within_equivalence_margin"] = 1 if abs(s["mean"]) < sesoi and lo > -sesoi and hi < sesoi else 0
    return out


def sesoi_band_checks(batch_rows, bands=(0.005, 0.010, 0.020)):
    rows = []
    for metric in ["selected_synthetic_utility", "selected_oracle_regret"]:
        for band in bands:
            result = paired_batch_effects(batch_rows, "full_framework", "direct_llm", metric, sesoi=band)
            rows.append(
                {
                    "metric": metric,
                    "sesoi_band": band,
                    "mean_diff": result["mean_diff"],
                    "empirical_p025": result["empirical_p025"],
                    "empirical_p975": result["empirical_p975"],
                    "within_sesoi_band": result["within_equivalence_margin"],
                }
            )
    return rows


def run_factorial_interactions(scenarios):
    global seed
    factors = ["evidence", "role", "flow", "critique", "gate"]
    base_low = {
        "label": "factorial",
        "evidence_capture": 0.38,
        "critique_rate": 0.22,
        "gate_strictness": 0.18,
        "role_boundedness": 0.30,
        "flow_control": 0.36,
        "load_factor": 1.08,
    }
    high = {
        "evidence": ("evidence_capture", 0.86),
        "role": ("role_boundedness", 0.86),
        "flow": ("flow_control", 0.86),
        "critique": ("critique_rate", 0.82),
        "gate": ("gate_strictness", 0.90),
    }
    design_rows = []
    seed_effect_rows = []
    seed_count = CONFIG.get("factorial_seed_count", 10)
    scenarios_per_seed = CONFIG.get("factorial_scenarios_per_seed", 1000)
    scenarios_subset = scenarios[:scenarios_per_seed]
    base_seed = CONFIG.get("factorial_seed_base", CONFIG.get("test_seed", 20262710) + 5000)

    terms = [
        ("evidence",),
        ("role",),
        ("flow",),
        ("critique",),
        ("gate",),
        ("evidence", "critique"),
        ("role", "gate"),
        ("critique", "gate"),
        ("evidence", "flow"),
        ("role", "flow", "gate"),
    ]

    def effect(design_subset, term):
        plus = []
        minus = []
        for row in design_subset:
            sign = 1
            for factor in term:
                sign *= 1 if row[factor] else -1
            (plus if sign > 0 else minus).append(row["selected_gpi_mean"])
        return sum(plus) / len(plus) - sum(minus) / len(minus)

    for seed_idx in range(seed_count):
        common_seed = base_seed + seed_idx * 37
        seed_design_rows = []
        for mask in range(2 ** len(factors)):
            policy = dict(base_low)
            levels = {}
            for idx, factor in enumerate(factors):
                on = 1 if mask & (1 << idx) else 0
                levels[factor] = on
                if on:
                    key, value = high[factor]
                    policy[key] = value
            policy_key = "f" + "".join(str(levels[factor]) for factor in factors)
            policy["label"] = policy_key
            seed = common_seed
            vals = [run_policy(scenario, policy_key, policy, PARAMS) for scenario in scenarios_subset]
            agg = aggregate(vals, ["policy", "policy_label"], ["selected_gpi", "selected_controllability", "selected_traceability"])[0]
            row = {
                "factorial_seed_index": seed_idx + 1,
                "factorial_seed": common_seed,
                **levels,
                **agg,
            }
            design_rows.append(row)
            seed_design_rows.append(row)
        for term in terms:
            seed_effect_rows.append(
                {
                    "factorial_seed_index": seed_idx + 1,
                    "factorial_seed": common_seed,
                    "term": " x ".join(term),
                    "gpi_effect": effect(seed_design_rows, term),
                }
            )

    effect_rows = []
    for term in [" x ".join(item) for item in terms]:
        vals = [row["gpi_effect"] for row in seed_effect_rows if row["term"] == term]
        s = stats(vals)
        sorted_vals = sorted(vals)
        lo = sorted_vals[max(0, math.floor(0.025 * (len(sorted_vals) - 1)))]
        hi = sorted_vals[min(len(sorted_vals) - 1, math.floor(0.975 * (len(sorted_vals) - 1)))]
        effect_rows.append(
            {
                "term": term,
                "seed_count": len(vals),
                "gpi_effect": s["mean"],
                "gpi_effect_sd": s["sd"],
                "gpi_effect_p025": lo,
                "gpi_effect_p975": hi,
            }
        )
    return design_rows, effect_rows


def rank(values):
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    out = [0] * len(values)
    for pos, (idx, _value) in enumerate(ordered):
        out[idx] = pos + 1
    return out


def correlation_diagnostics(rows):
    x = [row["selected_gpi"] for row in rows]
    y = [row["alternative_rule_score"] for row in rows]
    sx = stats(x)
    sy = stats(y)
    cov = sum((a - sx["mean"]) * (b - sy["mean"]) for a, b in zip(x, y)) / max(1, len(x) - 1)
    pearson = cov / max(1e-12, sx["sd"] * sy["sd"])
    rx = rank(x)
    ry = rank(y)
    srx = stats(rx)
    sry = stats(ry)
    rcov = sum((a - srx["mean"]) * (b - sry["mean"]) for a, b in zip(rx, ry)) / max(1, len(rx) - 1)
    spearman = rcov / max(1e-12, srx["sd"] * sry["sd"])
    by_scenario = {}
    for row in rows:
        by_scenario.setdefault(row["scenario_id"], []).append(row)
    top_agree = 0
    full_top_gpi = 0
    full_top_alt = 0
    for items in by_scenario.values():
        gpi_top = max(items, key=lambda item: item["selected_gpi"])["policy"]
        alt_top = max(items, key=lambda item: item["alternative_rule_score"])["policy"]
        top_agree += 1 if gpi_top == alt_top else 0
        full_top_gpi += 1 if gpi_top == "full_framework" else 0
        full_top_alt += 1 if alt_top == "full_framework" else 0
    n = len(by_scenario)
    return [
        {
            "pearson_gpi_alt": pearson,
            "spearman_gpi_alt": spearman,
            "scenario_top1_agreement": top_agree / n,
            "full_top1_by_gpi": full_top_gpi / n,
            "full_top1_by_alt": full_top_alt / n,
            "scenarios": n,
            "observations": len(rows),
        }
    ]


def run_scalability():
    results = []
    for n in [1000, 2500, 5000, 10000, 20000]:
        t0 = time.perf_counter()
        scenarios, rows = run_experiment(n)
        runtime_ms = (time.perf_counter() - t0) * 1000
        claim_records = sum(len(scenario["claims"]) for scenario in scenarios)
        results.append(
            {
                "scenarios": n,
                "claim_records": claim_records,
                "observations": len(rows),
                "runtime_ms": runtime_ms,
                "ms_per_scenario": runtime_ms / n,
            }
        )
    return results


def main():
    test_scenarios = CONFIG.get("test_scenarios", 10000)
    scenarios, rows = run_experiment(test_scenarios)
    scenario_rows, claim_rows, candidate_rows = scenario_record_tables(scenarios)
    metrics = [
        "selected_gpi",
        "selected_synthetic_utility",
        "selected_traceability",
        "selected_robustness",
        "selected_controllability",
        "cognitive_load",
    ]
    extended_metrics = metrics + [
        "selected_oracle_regret",
        "oracle_best_hit",
        "oracle_top2_hit",
        "evidence_coverage",
        "verified_coverage",
        "critique_coverage",
        "critique_effective",
        "severe_issue_acceptance_ratio",
        "issue_severity",
        "authority_violations",
        "low_reliability_verified_claim_proportion",
        "non_verified_linked_claim_ratio",
        "unresolved_risk_rate",
        "source_diversity",
        "selected_risk_penalty",
        "alternative_rule_score",
    ]
    summary = aggregate(rows, ["policy", "policy_label"], extended_metrics)
    public_method_rows = run_public_method_comparison(scenarios)
    public_method_summary = aggregate(public_method_rows, ["method", "method_label", "reference"], extended_metrics)
    public_method_tests = [
        paired_test(public_method_rows, "ours", "coa_gpt_style", "selected_gpi"),
        paired_test(public_method_rows, "ours", "rag_coa_style", "selected_gpi"),
        paired_test(public_method_rows, "ours", "autogen_style", "selected_gpi"),
        paired_test(public_method_rows, "ours", "masmp_style", "selected_gpi"),
        paired_test(public_method_rows, "ours", "coa_gpt_style", "selected_synthetic_utility"),
        paired_test(public_method_rows, "ours", "rag_coa_style", "selected_synthetic_utility"),
        paired_test(public_method_rows, "ours", "autogen_style", "selected_synthetic_utility"),
        paired_test(public_method_rows, "ours", "masmp_style", "selected_synthetic_utility"),
        paired_test(public_method_rows, "ours", "coa_gpt_style", "selected_traceability"),
        paired_test(public_method_rows, "ours", "coa_gpt_style", "selected_controllability"),
        paired_test(public_method_rows, "ours", "coa_gpt_style", "selected_oracle_regret"),
    ]
    ablation_rows = run_ablation(scenarios)
    ablation_summary = aggregate(ablation_rows, ["policy", "policy_label"], extended_metrics)
    profile_summary = aggregate(
        [row for row in rows if row["policy"] in {"direct_llm", "full_framework"}],
        ["profile", "policy", "policy_label"],
        ["selected_gpi", "selected_synthetic_utility", "selected_oracle_regret", "oracle_best_hit", "oracle_top2_hit", "selected_traceability", "selected_controllability"],
    )
    sensitivity_rows = run_sensitivity(scenarios)
    sensitivity_summary = aggregate(
        sensitivity_rows,
        ["setting"],
        [
            "selected_gpi",
            "selected_synthetic_utility",
            "selected_traceability",
            "selected_controllability",
            "cognitive_load",
            "low_reliability_verified_claim_proportion",
            "non_verified_linked_claim_ratio",
            "unresolved_risk_rate",
            "selected_risk_penalty",
        ],
    )
    sensitivity_seed_summary = aggregate(
        sensitivity_rows,
        ["setting", "sensitivity_seed_index", "sensitivity_seed"],
        [
            "selected_gpi",
            "selected_synthetic_utility",
            "selected_traceability",
            "selected_controllability",
            "cognitive_load",
            "low_reliability_verified_claim_proportion",
            "non_verified_linked_claim_ratio",
            "selected_risk_penalty",
        ],
    )
    selected_by_seed = {
        row["sensitivity_seed_index"]: row
        for row in sensitivity_seed_summary
        if row["setting"] == "selected"
    }
    sensitivity_paired_deltas = []
    for row in sensitivity_seed_summary:
        baseline = selected_by_seed.get(row["sensitivity_seed_index"])
        if row["setting"] == "selected" or not baseline:
            continue
        sensitivity_paired_deltas.append(
            {
                "setting": row["setting"],
                "sensitivity_seed_index": row["sensitivity_seed_index"],
                "sensitivity_seed": row["sensitivity_seed"],
                "delta_gpi_vs_selected": row["selected_gpi_mean"] - baseline["selected_gpi_mean"],
                "delta_low_reliability_verified_vs_selected": row["low_reliability_verified_claim_proportion_mean"]
                - baseline["low_reliability_verified_claim_proportion_mean"],
                "delta_non_verified_linked_claim_vs_selected": row["non_verified_linked_claim_ratio_mean"]
                - baseline["non_verified_linked_claim_ratio_mean"],
            }
        )
    seed_batch_rows = run_seed_batches()
    seed_batch_tests = [
        paired_batch_effects(seed_batch_rows, "full_framework", "direct_llm", "selected_gpi"),
        paired_batch_effects(seed_batch_rows, "full_framework", "direct_llm", "selected_synthetic_utility", sesoi=0.01),
        paired_batch_effects(seed_batch_rows, "full_framework", "direct_llm", "selected_oracle_regret", sesoi=0.01),
        paired_batch_effects(seed_batch_rows, "full_framework", "evidence_only", "selected_gpi"),
        paired_batch_effects(seed_batch_rows, "full_framework", "evidence_critique", "selected_gpi"),
    ]
    sesoi_checks = sesoi_band_checks(seed_batch_rows)
    factorial_design, factorial_effects = run_factorial_interactions(scenarios)
    corr_diagnostics = correlation_diagnostics(rows)
    scalability = run_scalability()
    tests = [
        paired_test(rows, "full_framework", "direct_llm", "selected_gpi"),
        paired_test(rows, "full_framework", "evidence_only", "selected_gpi"),
        paired_test(rows, "full_framework", "evidence_role", "selected_gpi"),
        paired_test(rows, "full_framework", "evidence_critique", "selected_gpi"),
        paired_test(rows, "full_framework", "direct_llm", "selected_synthetic_utility"),
        paired_test(rows, "full_framework", "evidence_only", "selected_synthetic_utility"),
        paired_test(rows, "full_framework", "direct_llm", "selected_oracle_regret"),
        paired_test(rows, "full_framework", "direct_llm", "selected_traceability"),
        paired_test(rows, "full_framework", "direct_llm", "selected_controllability"),
        paired_test(rows, "full_framework", "direct_llm", "alternative_rule_score"),
    ]

    write_csv(OUT / "experiment_policy_results.csv", rows, list(rows[0].keys()))
    write_csv(OUT / "scenario_records.csv", scenario_rows, list(scenario_rows[0].keys()))
    write_csv(OUT / "scenario_claims.csv", claim_rows, list(claim_rows[0].keys()))
    write_csv(OUT / "candidate_coas.csv", candidate_rows, list(candidate_rows[0].keys()))
    write_csv(OUT / "experiment_comparison_summary.csv", summary, list(summary[0].keys()))
    write_csv(OUT / "experiment_public_method_results.csv", public_method_rows, list(public_method_rows[0].keys()))
    write_csv(OUT / "experiment_public_method_summary.csv", public_method_summary, list(public_method_summary[0].keys()))
    write_csv(OUT / "experiment_public_method_stat_tests.csv", public_method_tests, list(public_method_tests[0].keys()))
    write_csv(OUT / "experiment_ablation_results.csv", ablation_rows, list(ablation_rows[0].keys()))
    write_csv(OUT / "experiment_ablation_summary.csv", ablation_summary, list(ablation_summary[0].keys()))
    write_csv(OUT / "experiment_profile_summary.csv", profile_summary, list(profile_summary[0].keys()))
    write_csv(OUT / "experiment_sensitivity_summary.csv", sensitivity_summary, list(sensitivity_summary[0].keys()))
    write_csv(OUT / "experiment_sensitivity_seed_summary.csv", sensitivity_seed_summary, list(sensitivity_seed_summary[0].keys()))
    write_csv(
        OUT / "experiment_sensitivity_paired_deltas.csv",
        sensitivity_paired_deltas,
        list(sensitivity_paired_deltas[0].keys()),
    )
    write_csv(OUT / "experiment_seed_batch_summary.csv", seed_batch_rows, list(seed_batch_rows[0].keys()))
    write_csv(OUT / "experiment_seed_batch_tests.csv", seed_batch_tests, list(seed_batch_tests[0].keys()))
    write_csv(OUT / "experiment_sesoi_band_checks.csv", sesoi_checks, list(sesoi_checks[0].keys()))
    write_csv(OUT / "experiment_factorial_design.csv", factorial_design, list(factorial_design[0].keys()))
    write_csv(OUT / "experiment_factorial_effects.csv", factorial_effects, list(factorial_effects[0].keys()))
    write_csv(OUT / "experiment_alternative_audit_diagnostics.csv", corr_diagnostics, list(corr_diagnostics[0].keys()))
    write_csv(OUT / "experiment_stat_tests.csv", tests, list(tests[0].keys()))
    write_csv(OUT / "experiment_scalability.csv", scalability, list(scalability[0].keys()))
    write_csv(OUT / "experiment_full_metric_summary.csv", summary, list(summary[0].keys()))

    make_figures(summary, rows, ablation_summary, profile_summary, scalability)
    make_public_method_figure(public_method_summary)

    find = lambda policy: next(item for item in summary if item["policy"] == policy)
    full = find("full_framework")
    direct = find("direct_llm")
    evidence_only = find("evidence_only")
    rel = lambda a, b, key: (a[key] - b[key]) / max(1e-12, b[key]) * 100

    report = [
        "# Experimental Results",
        "",
        "## Reproducible Run",
        "",
        "- Script: `code/c2_experiments.py`",
        f"- Scenario count: {test_scenarios:,}",
        f"- Claim records: {len(claim_rows):,}",
        f"- Candidate COA records: {len(candidate_rows):,}",
        f"- Policy observations: {len(rows):,}",
        f"- Ablation observations: {len(ablation_rows):,}",
        f"- Sensitivity observations: {len(sensitivity_rows):,}",
        f"- Sensitivity common-random-number seeds: {CONFIG.get('sensitivity_seed_count', 20)}",
        f"- Factorial common-random-number seeds: {CONFIG.get('factorial_seed_count', 10)}",
        "- Scenario profiles: anti-UAV, open event, humanitarian C2",
        "",
        "## Public Method-Family Proxy Comparison",
        "",
        "| Method | Reference | GPI | Synthetic utility | Oracle regret | Top-2 hit | Traceability | Robustness | Controllability | Cognitive load |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in public_method_summary:
        report.append(
            f"| {row['method_label']} | {row['reference']} | {row['selected_gpi_mean']:.3f} | "
            f"{row['selected_synthetic_utility_mean']:.3f} | {row['selected_oracle_regret_mean']:.3f} | "
            f"{row['oracle_top2_hit_mean']:.3f} | "
            f"{row['selected_traceability_mean']:.3f} | {row['selected_robustness_mean']:.3f} | "
            f"{row['selected_controllability_mean']:.3f} | {row['cognitive_load_mean']:.3f} |"
        )
    report.extend(
        [
            "",
        "## Main Comparison",
        "",
        "| Policy | GPI | Synthetic utility | Oracle regret | Oracle-best hit | Top-2 hit | Traceability | Robustness | Controllability | Cognitive load | Alternative audit |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary:
        report.append(
            f"| {row['policy_label']} | {row['selected_gpi_mean']:.3f} | "
            f"{row['selected_synthetic_utility_mean']:.3f} | {row['selected_oracle_regret_mean']:.3f} | "
            f"{row['oracle_best_hit_mean']:.3f} | {row['oracle_top2_hit_mean']:.3f} | "
            f"{row['selected_traceability_mean']:.3f} | {row['selected_robustness_mean']:.3f} | "
            f"{row['selected_controllability_mean']:.3f} | {row['cognitive_load_mean']:.3f} | "
            f"{row['alternative_rule_score_mean']:.3f} |"
        )
    report.extend(
        [
            "",
            (
                "Full framework improved mean GPI by "
                f"{rel(full, direct, 'selected_gpi_mean'):.1f}% over Direct-generation proxy and by "
                f"{rel(full, evidence_only, 'selected_gpi_mean'):.1f}% over Evidence only. "
                f"It also increased traceability by {rel(full, direct, 'selected_traceability_mean'):.1f}% "
                f"and controllability by {rel(full, direct, 'selected_controllability_mean'):.1f}% "
                "relative to Direct-generation proxy."
            ),
            "",
            "## Reproducibility",
            "",
            "```powershell",
            "& 'C:\\Users\\85184\\AppData\\Local\\Programs\\Python\\Python39\\python.exe' code\\c2_experiments.py",
            "```",
            "",
        ]
    )
    (REPORT / "EXPERIMENT_RESULTS_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "scenarios": len(scenarios),
                "observations": len(rows),
                "summary": [
                    {
                        "policy": item["policy"],
                        "gpi": item["selected_gpi_mean"],
                        "synthetic_utility": item["selected_synthetic_utility_mean"],
                        "oracle_regret": item["selected_oracle_regret_mean"],
                        "traceability": item["selected_traceability_mean"],
                        "controllability": item["selected_controllability_mean"],
                    }
                    for item in summary
                ],
                "scalability": scalability,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
