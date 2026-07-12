from __future__ import annotations

import csv
import json
from pathlib import Path

import c2_experiments as exp

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "calibration_config.json"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

L9_LEVELS = [
    (1, 1, 1, 1),
    (1, 2, 2, 2),
    (1, 3, 3, 3),
    (2, 1, 2, 3),
    (2, 2, 3, 1),
    (2, 3, 1, 2),
    (3, 1, 3, 2),
    (3, 2, 1, 3),
    (3, 3, 2, 1),
]


def read_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_csv(path: Path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return sum(values) / len(values)


def objective(row):
    return (
        0.24 * row["selected_traceability"]
        + 0.20 * row["severe_issue_acceptance_ratio"]
        + 0.20 * row["selected_controllability"]
        + 0.16 * row["selected_robustness"]
        - 0.10 * row["cognitive_load"]
        - 0.06 * row["low_reliability_verified_claim_proportion"]
        - 0.04 * row["unresolved_risk_rate"]
    )


def evaluate_setting(scenarios, theta_e, p_q, p_g, lambda_l):
    params = {"theta_e": theta_e, "lambda_l": lambda_l}
    policy = {
        **exp.POLICIES["full_framework"],
        "critique_rate": p_q,
        "gate_strictness": p_g,
    }
    rows = [exp.run_policy(scenario, "full_framework", policy, params) for scenario in scenarios]
    return {
        "J_p": mean([objective(row) for row in rows]),
        "traceability": mean([row["selected_traceability"] for row in rows]),
        "critique_effectiveness": mean([row["severe_issue_acceptance_ratio"] for row in rows]),
        "controllability": mean([row["selected_controllability"] for row in rows]),
        "robustness": mean([row["selected_robustness"] for row in rows]),
        "cognitive_load": mean([row["cognitive_load"] for row in rows]),
        "low_reliability_verified_claim_proportion": mean(
            [row["low_reliability_verified_claim_proportion"] for row in rows]
        ),
        "unresolved_risk_rate": mean([row["unresolved_risk_rate"] for row in rows]),
    }


def main():
    cfg = read_config()
    factors = cfg["factors"]

    exp.seed = cfg["calibration_seed"]
    calibration_scenarios = exp.generate_scenarios(cfg["calibration_scenarios"])
    exp.seed = cfg["validation_seed"]
    validation_scenarios = exp.generate_scenarios(cfg["validation_scenarios"])

    rows = []
    for row_id, levels in enumerate(L9_LEVELS, start=1):
        theta_level, pq_level, pg_level, lambda_level = levels
        theta_e = factors["theta_e"][theta_level - 1]
        p_q = factors["p_q"][pq_level - 1]
        p_g = factors["p_g"][pg_level - 1]
        lambda_l = factors["lambda_l"][lambda_level - 1]
        start = (row_id - 1) * cfg["calibration_scenarios_per_l9_row"]
        end = start + cfg["calibration_scenarios_per_l9_row"]

        exp.seed = cfg["calibration_seed"] + 100 + row_id
        calibration = evaluate_setting(calibration_scenarios[start:end], theta_e, p_q, p_g, lambda_l)
        exp.seed = cfg["validation_seed"] + 100
        validation = evaluate_setting(validation_scenarios, theta_e, p_q, p_g, lambda_l)
        rows.append(
            {
                "row": row_id,
                "theta_e_level": theta_level,
                "p_q_level": pq_level,
                "p_g_level": pg_level,
                "lambda_l_level": lambda_level,
                "theta_e": theta_e,
                "p_q": p_q,
                "p_g": p_g,
                "lambda_l": lambda_l,
                "calibration_J_p": calibration["J_p"],
                "validation_J_p": validation["J_p"],
                "calibration_traceability": calibration["traceability"],
                "calibration_critique_effectiveness": calibration["critique_effectiveness"],
                "calibration_controllability": calibration["controllability"],
                "calibration_robustness": calibration["robustness"],
                "calibration_cognitive_load": calibration["cognitive_load"],
                "calibration_low_reliability_verified_claim_proportion": calibration[
                    "low_reliability_verified_claim_proportion"
                ],
                "calibration_unresolved_risk_rate": calibration["unresolved_risk_rate"],
                "validation_traceability": validation["traceability"],
                "validation_critique_effectiveness": validation["critique_effectiveness"],
                "validation_controllability": validation["controllability"],
                "validation_robustness": validation["robustness"],
                "validation_cognitive_load": validation["cognitive_load"],
                "validation_low_reliability_verified_claim_proportion": validation[
                    "low_reliability_verified_claim_proportion"
                ],
                "validation_unresolved_risk_rate": validation["unresolved_risk_rate"],
            }
        )

    ranked = sorted(rows, key=lambda item: item["validation_J_p"], reverse=True)
    prior_value = None
    prior_rank = 0
    for position, row in enumerate(ranked, start=1):
        rank = prior_rank if prior_value is not None and abs(row["validation_J_p"] - prior_value) < 1e-12 else position
        prior_rank = rank
        prior_value = row["validation_J_p"]
        row["validation_rank"] = rank
        row["validation_margin_to_best"] = ranked[0]["validation_J_p"] - row["validation_J_p"]
        row["validation_margin_to_second"] = (
            row["validation_J_p"] - ranked[1]["validation_J_p"] if rank == 1 and len(ranked) > 1 else ""
        )

    range_rows = []
    factor_map = {
        "theta_e": ("theta_e_level", "theta_e"),
        "p_q": ("p_q_level", "p_q"),
        "p_g": ("p_g_level", "p_g"),
        "lambda_l": ("lambda_l_level", "lambda_l"),
    }
    for factor, (level_key, value_key) in factor_map.items():
        level_means = []
        for level in (1, 2, 3):
            vals = [row["calibration_J_p"] for row in rows if row[level_key] == level]
            value = [row[value_key] for row in rows if row[level_key] == level][0]
            level_means.append((level, value, mean(vals)))
        best_level, selected_value, best_mean = max(level_means, key=lambda item: item[2])
        range_rows.append(
            {
                "factor": factor,
                "level1_mean": level_means[0][2],
                "level2_mean": level_means[1][2],
                "level3_mean": level_means[2][2],
                "range": max(item[2] for item in level_means) - min(item[2] for item in level_means),
                "best_level": best_level,
                "selected_value": selected_value,
            }
        )

    selected = cfg["selected_parameters"]
    exp.seed = cfg["calibration_seed"] + 999
    selected_calibration = evaluate_setting(
        calibration_scenarios, selected["theta_e"], selected["p_q"], selected["p_g"], selected["lambda_l"]
    )
    exp.seed = cfg["validation_seed"] + 100
    selected_validation = evaluate_setting(
        validation_scenarios, selected["theta_e"], selected["p_q"], selected["p_g"], selected["lambda_l"]
    )
    selected_row = {
        "row": "range_selected",
        "theta_e_level": 2,
        "p_q_level": 2,
        "p_g_level": 2,
        "lambda_l_level": 1,
        "theta_e": selected["theta_e"],
        "p_q": selected["p_q"],
        "p_g": selected["p_g"],
        "lambda_l": selected["lambda_l"],
        "calibration_J_p": selected_calibration["J_p"],
        "validation_J_p": selected_validation["J_p"],
        "calibration_traceability": selected_calibration["traceability"],
        "calibration_critique_effectiveness": selected_calibration["critique_effectiveness"],
        "calibration_controllability": selected_calibration["controllability"],
        "calibration_robustness": selected_calibration["robustness"],
        "calibration_cognitive_load": selected_calibration["cognitive_load"],
        "calibration_low_reliability_verified_claim_proportion": selected_calibration[
            "low_reliability_verified_claim_proportion"
        ],
        "calibration_unresolved_risk_rate": selected_calibration["unresolved_risk_rate"],
        "validation_traceability": selected_validation["traceability"],
        "validation_critique_effectiveness": selected_validation["critique_effectiveness"],
        "validation_controllability": selected_validation["controllability"],
        "validation_robustness": selected_validation["robustness"],
        "validation_cognitive_load": selected_validation["cognitive_load"],
        "validation_low_reliability_verified_claim_proportion": selected_validation[
            "low_reliability_verified_claim_proportion"
        ],
        "validation_unresolved_risk_rate": selected_validation["unresolved_risk_rate"],
    }
    validation_ranked = sorted(rows + [selected_row], key=lambda item: item["validation_J_p"], reverse=True)
    prior_value = None
    prior_rank = 0
    for position, row in enumerate(validation_ranked, start=1):
        rank = prior_rank if prior_value is not None and abs(row["validation_J_p"] - prior_value) < 1e-12 else position
        prior_rank = rank
        prior_value = row["validation_J_p"]
        row["validation_rank"] = rank
        row["validation_margin_to_best"] = validation_ranked[0]["validation_J_p"] - row["validation_J_p"]
        row["validation_margin_to_second"] = (
            row["validation_J_p"] - validation_ranked[1]["validation_J_p"] if rank == 1 and len(validation_ranked) > 1 else ""
        )

    result_columns = list(validation_ranked[0].keys())
    write_csv(OUT / "l9_calibration_results.csv", rows, result_columns)
    write_csv(OUT / "l9_validation_results.csv", validation_ranked, result_columns)
    write_csv(OUT / "l9_range_analysis.csv", range_rows, list(range_rows[0].keys()))
    print(
        json.dumps(
            {
                "best_validation_row": validation_ranked[0]["row"],
                "best_validation_J_p": validation_ranked[0]["validation_J_p"],
                "second_validation_J_p": validation_ranked[1]["validation_J_p"],
                "margin_to_second": validation_ranked[0]["validation_J_p"] - validation_ranked[1]["validation_J_p"],
                "range_selected_rank": selected_row["validation_rank"],
                "range_selected_J_p": selected_row["validation_J_p"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
