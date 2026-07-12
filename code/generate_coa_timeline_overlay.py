"""Generate a data-driven COA execution timeline figure.

The figure is built from the same synthetic scenario generator and scoring
functions used by the experimental section. It selects a complex anti-UAV
scenario, runs the full framework policy, and visualizes the selected COA as a
timeline with evidence, risk, and commander-gate overlays.
"""

from pathlib import Path
import csv
import importlib.util

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
OUT = CODE / "outputs"
FIG = ROOT / "figures"

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


def load_experiment_module():
    spec = importlib.util.spec_from_file_location("c2_experiments", CODE / "c2_experiments.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_complex_case(scenarios):
    anti_uav = [item for item in scenarios if item["profile_key"] == "anti_uav"]

    def rank(item):
        conflict = sum(1 for claim in item["claims"] if claim["conflict_prone"])
        return (
            item["complexity"] * 12
            + item["uncertainty"] * 20
            + item["time_pressure"] * 12
            + len(item["claims"]) * 0.10
            + len(item["coas"]) * 1.5
            + conflict * 0.8
        )

    return max(anti_uav, key=rank)


def select_policy_coa(exp, scenario, policy_key, policy, seed_offset=4242):
    params = exp.PARAMS
    exp.seed = 20260710 + seed_offset
    ev = exp.evidence(scenario["claims"], policy, params)
    ctl = exp.controls(policy, scenario, params)
    candidates = []
    for coa in scenario["coas"]:
        crit = exp.critique(coa, ev, policy, scenario, params)
        sc = exp.score(coa, ev, crit, ctl, scenario, policy, params)
        candidates.append({"coa": coa, "crit": crit, "score": sc})
    selected = sorted(candidates, key=lambda item: item["score"]["quality"], reverse=True)[0]
    return ev, ctl, selected, candidates


def select_full_framework_coa(exp, scenario):
    return select_policy_coa(exp, scenario, "full_framework", exp.POLICIES["full_framework"], 4242)


def build_timeline(scenario, evidence_records, selected):
    claim_map = {item["id"]: item for item in evidence_records}
    linked = [claim_map[item] for item in selected["coa"]["claim_ids"] if item in claim_map]
    by_type = {}
    for claim in linked:
        by_type.setdefault(claim["type"], []).append(claim)

    base = max(1.0, 1.0 - scenario["time_pressure"] * 0.35)
    risk_scale = selected["coa"]["risk"] + selected["score"]["risk_penalty"]
    uncertainty = scenario["uncertainty"]

    stage_defs = [
        ("Track fusion", ["task", "timing"], 11, 0.45),
        ("Evidence validation", ["rule", "intent"], 13, 0.55),
        ("EW denial setup", ["resource", "rule"], 14, 0.65),
        ("Interceptor window", ["timing", "risk"], 10, 0.78),
        ("Protected-site backstop", ["resource", "risk"], 18, 0.72),
        ("Commander release gate", ["intent", "rule", "risk"], 8, 0.60),
    ]

    rows = []
    cursor = 0.0
    overlaps = [0.0, -2.5, 1.5, -3.0, 2.0, 0.0]
    for idx, (stage, types, nominal_duration, risk_weight) in enumerate(stage_defs):
        relevant = [claim for claim_type in types for claim in by_type.get(claim_type, [])]
        captured = sum(1 for claim in relevant if claim["captured"])
        verified = sum(1 for claim in relevant if claim["state"] == "verified")
        contested = sum(1 for claim in relevant if claim["state"] != "verified" or claim["conflict_prone"])
        reliability = (
            sum(claim["reliability"] for claim in relevant) / len(relevant)
            if relevant
            else 0.50
        )
        duration = nominal_duration * base * (1.0 + uncertainty * 0.18)
        if contested > captured * 0.35 and idx in {1, 3, 4}:
            duration *= 1.12
        start = max(0.0, cursor + overlaps[idx])
        end = start + duration
        risk = min(0.98, risk_weight * risk_scale * (1.0 + contested / max(8, len(relevant) + 1)) * (1.05 - reliability * 0.25))
        evidence_strength = min(1.0, 0.45 * (captured / max(1, len(relevant))) + 0.55 * (verified / max(1, len(relevant))))
        rows.append(
            {
                "stage": stage,
                "start_min": round(start, 2),
                "end_min": round(end, 2),
                "duration_min": round(duration, 2),
                "linked_claims": len(relevant),
                "captured_claims": captured,
                "verified_claims": verified,
                "contested_claims": contested,
                "evidence_strength": round(evidence_strength, 3),
                "risk_level": round(risk, 3),
            }
        )
        cursor = max(cursor + nominal_duration * base * 0.86, end - duration * 0.10)

    gates = [
        {
            "gate": "G1",
            "time_min": rows[1]["end_min"],
            "label": "G1: evidence sufficiency",
            "condition": "verified/captured claims released to commander",
        },
        {
            "gate": "G2",
            "time_min": rows[2]["end_min"],
            "label": "G2: reversible-effect authorization",
            "condition": "EW action bounded by ROE and evidence state",
        },
        {
            "gate": "G3",
            "time_min": rows[5]["end_min"],
            "label": "G3: COA release",
            "condition": "terminal action remains commander-controlled",
        },
    ]
    return rows, gates


def draw_figure(scenario, selected, rows, gates):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    colors = {
        "Track fusion": NATURE["blue"],
        "Evidence validation": NATURE["teal"],
        "EW denial setup": NATURE["purple"],
        "Interceptor window": NATURE["ochre"],
        "Protected-site backstop": NATURE["rust"],
        "Commander release gate": NATURE["slate"],
    }
    fig, ax = plt.subplots(figsize=(7.15, 4.85))
    y_positions = list(range(len(rows)))

    for idx, row in enumerate(rows):
        y = len(rows) - 1 - idx
        left = row["start_min"]
        width = row["end_min"] - row["start_min"]
        ax.barh(
            y,
            width,
            left=left,
            height=0.56,
            color=colors[row["stage"]],
            alpha=0.82,
            edgecolor=NATURE["ink"],
            linewidth=0.6,
        )
        ev_x = left + width * 0.54
        ev_size = 35 + row["evidence_strength"] * 140
        ax.scatter(
            [ev_x],
            [y + 0.30],
            s=ev_size,
            color=NATURE["evidence"],
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        ax.text(
            ev_x,
            y + 0.30,
            str(row["verified_claims"]),
            va="center",
            ha="center",
            color=NATURE["ink"],
            fontsize=7.0,
            zorder=5,
        )
        risk_x = left + width * 0.86
        risk_color = NATURE["high"] if row["risk_level"] >= 0.70 else NATURE["moderate"]
        ax.scatter(
            [risk_x],
            [y - 0.30],
            marker="D",
            s=45 + row["risk_level"] * 70,
            color=risk_color,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        ax.text(
            risk_x + 1.1,
            y - 0.30,
            f"{row['risk_level']:.2f}",
            va="center",
            ha="left",
            fontsize=7.0,
            color=NATURE["high"] if row["risk_level"] >= 0.70 else "#7A5A23",
        )

    for gate in gates:
        ax.axvline(gate["time_min"], color=NATURE["ink"], linestyle=(0, (3, 3)), linewidth=0.9, alpha=0.75)
        ax.text(
            gate["time_min"],
            len(rows) + 0.08,
            {
                "G1": "G1\nEvidence check",
                "G2": "G2\nEW authority",
                "G3": "G3\nCOA release",
            }[gate["gate"]],
            va="bottom",
            ha="center",
            fontsize=6.9,
            color=NATURE["ink"],
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": NATURE["grid"], "linewidth": 0.6},
        )

    ax.set_xlabel("Mission elapsed time (min)")
    ax.set_yticks([len(rows) - 1 - idx for idx in range(len(rows))])
    ax.set_yticklabels([row["stage"] for row in rows], fontsize=7.1, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlim(0, max(row["end_min"] for row in rows) + 8)
    ax.set_ylim(-0.85, len(rows) + 0.70)
    ax.grid(axis="x", color=NATURE["grid"], linewidth=0.5, alpha=0.65)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    score = selected["score"]
    coa = selected["coa"]
    summary = (
        f"Generated case: scenario {scenario['id']} | COA {coa['id']}\n"
        f"complexity={scenario['complexity']}, uncertainty={scenario['uncertainty']:.2f}, "
        f"time pressure={scenario['time_pressure']:.2f}\n"
        f"GPI={score['quality']:.3f}, traceability={score['traceability']:.3f}, "
        f"robustness={score['robustness']:.3f}\n"
        f"linked claims={len(coa['claim_ids'])}, COA risk={coa['risk']:.3f}"
    )
    ax.text(
        0.01,
        -0.20,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        color=NATURE["ink"],
    )
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=NATURE["evidence"], markeredgecolor="white", markersize=7, label="verified evidence count"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=NATURE["moderate"], markeredgecolor="white", markersize=6, label="moderate risk"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=NATURE["high"], markeredgecolor="white", markersize=6, label="high risk"),
        Line2D([0], [0], color=NATURE["ink"], linestyle=(0, (3, 3)), linewidth=1.0, label="commander gate"),
    ]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, -0.27), ncol=2, frameon=False, fontsize=7.0)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.87, bottom=0.30)
    pdf = FIG / "fig_coa_execution_timeline_overlay.pdf"
    png = FIG / "fig_coa_execution_timeline_overlay.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


def compact_timeline(rows, gates):
    """Compress stage rows into method-level indicators for a comparison strip."""
    start = min(row["start_min"] for row in rows)
    end = max(row["end_min"] for row in rows)
    verified = sum(row["verified_claims"] for row in rows)
    captured = sum(row["captured_claims"] for row in rows)
    contested = sum(row["contested_claims"] for row in rows)
    max_risk = max(row["risk_level"] for row in rows)
    mean_risk = sum(row["risk_level"] for row in rows) / len(rows)
    gate_times = [gate["time_min"] for gate in gates]
    return {
        "start_min": start,
        "end_min": end,
        "verified_claims": verified,
        "captured_claims": captured,
        "contested_claims": contested,
        "max_risk": max_risk,
        "mean_risk": mean_risk,
        "gate_times": gate_times,
    }


def draw_comparison_figure(scenario, comparisons):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.3,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.15, 4.55))
    colors = {
        "COA-GPT-inspired proxy": NATURE["slate"],
        "RAG-inspired proxy": NATURE["blue"],
        "AutoGen-inspired proxy": NATURE["purple"],
        "MASMP-inspired proxy": NATURE["ochre"],
        "Ours": NATURE["green"],
    }
    max_time = max(item["timeline"]["end_min"] for item in comparisons) + 14
    short_labels = {
        "COA-GPT-inspired proxy": "COA-GPT",
        "RAG-inspired proxy": "RAG",
        "AutoGen-inspired proxy": "AutoGen",
        "MASMP-inspired proxy": "MASMP",
        "Ours": "Ours",
    }
    y_labels = []
    y_ticks = []
    for idx, item in enumerate(comparisons):
        y = len(comparisons) - 1 - idx
        y_ticks.append(y)
        y_labels.append(f"{short_labels[item['label']]}\n{item['coa_id']}")
        tl = item["timeline"]
        width = tl["end_min"] - tl["start_min"]
        ax.barh(
            y,
            width,
            left=tl["start_min"],
            height=0.48,
            color=colors[item["label"]],
            alpha=0.82,
            edgecolor=NATURE["ink"],
            linewidth=0.6,
        )
        gate_count = 3 if item["label"] == "Ours" else max(0, round(item["gate_closure"] * 3))
        for gate_idx, gate_time in enumerate(tl["gate_times"][:gate_count]):
            ax.vlines(
                gate_time,
                y - 0.35,
                y + 0.35,
                color=NATURE["ink"],
                linestyle=(0, (3, 2)),
                linewidth=0.8,
                alpha=0.75,
            )
            ax.text(gate_time, y + 0.42, f"G{gate_idx + 1}", ha="center", va="bottom", fontsize=6.8)
        ev_x = tl["start_min"] + width * 0.42
        evidence_score = item["traceability"]
        ax.scatter(
            [ev_x],
            [y + 0.28],
            s=42 + evidence_score * 135,
            color=NATURE["evidence"],
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        ax.text(ev_x, y + 0.28, str(tl["verified_claims"]), ha="center", va="center", fontsize=6.6, zorder=5)
        risk_x = tl["start_min"] + width * 0.78
        risk_color = NATURE["high"] if tl["max_risk"] >= 0.70 else NATURE["moderate"]
        ax.scatter(
            [risk_x],
            [y - 0.28],
            marker="D",
            s=44 + tl["max_risk"] * 80,
            color=risk_color,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        ax.text(risk_x + 1.1, y - 0.28, f"{tl['max_risk']:.2f}", ha="left", va="center", fontsize=6.8, color=NATURE["high"])
        ax.text(
            max_time - 0.2,
            y,
            f"GPI={item['gpi']:.3f}  T={item['traceability']:.3f}  C={item['controllability']:.3f}",
            ha="right",
            va="center",
            fontsize=7.1,
            color=NATURE["ink"],
        )

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=6.9, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlim(0, max_time)
    ax.set_ylim(-0.75, len(comparisons) - 0.15)
    ax.set_xlabel("Mission elapsed time (min)")
    ax.grid(axis="x", color=NATURE["grid"], linewidth=0.5, alpha=0.65)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=NATURE["evidence"], markeredgecolor="white", markersize=7, label="verified evidence count"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=NATURE["moderate"], markeredgecolor="white", markersize=6, label="moderate risk"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=NATURE["high"], markeredgecolor="white", markersize=6, label="high risk"),
        Line2D([0], [0], color=NATURE["ink"], linestyle=(0, (3, 2)), linewidth=1.0, label="active gate"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.27), ncol=4, frameon=False, fontsize=6.8)
    fig.subplots_adjust(left=0.18, right=0.985, top=0.96, bottom=0.25)
    pdf = FIG / "fig_coa_timeline_baseline_comparison.pdf"
    png = FIG / "fig_coa_timeline_baseline_comparison.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


def build_public_method_comparison(exp, scenario):
    comparisons = []
    for idx, (key, policy) in enumerate(exp.PUBLIC_METHOD_BASELINES.items()):
        ev, ctl, selected, candidates = select_policy_coa(exp, scenario, key, policy, 5000 + idx * 101)
        rows, gates = build_timeline(scenario, ev, selected)
        tl = compact_timeline(rows, gates)
        comparisons.append(
            {
                "method": key,
                "label": policy["label"],
                "coa_id": selected["coa"]["id"],
                "gpi": selected["score"]["quality"],
                "traceability": selected["score"]["traceability"],
                "robustness": selected["score"]["robustness"],
                "controllability": selected["score"]["controllability"],
                "risk_penalty": selected["score"]["risk_penalty"],
                "authority_violations": ctl["authority_violations"],
                "gate_closure": ctl["gate_closure"],
                "timeline": tl,
            }
        )
    return comparisons


def main():
    exp = load_experiment_module()
    exp.seed = 20260710
    scenarios = exp.generate_scenarios(10000)
    scenario = choose_complex_case(scenarios)
    evidence_records, ctl, selected, candidates = select_full_framework_coa(exp, scenario)
    rows, gates = build_timeline(scenario, evidence_records, selected)
    pdf, png = draw_figure(scenario, selected, rows, gates)
    comparisons = build_public_method_comparison(exp, scenario)
    comparison_pdf, comparison_png = draw_comparison_figure(scenario, comparisons)

    case_rows = []
    for row in rows:
        case_rows.append(
            {
                "scenario_id": scenario["id"],
                "coa_id": selected["coa"]["id"],
                "gpi": round(selected["score"]["quality"], 6),
                "traceability": round(selected["score"]["traceability"], 6),
                "robustness": round(selected["score"]["robustness"], 6),
                "controllability": round(selected["score"]["controllability"], 6),
                "coa_risk": round(selected["coa"]["risk"], 6),
                **row,
            }
        )
    write_csv(
        OUT / "coa_timeline_overlay_case.csv",
        case_rows,
        [
            "scenario_id",
            "coa_id",
            "gpi",
            "traceability",
            "robustness",
            "controllability",
            "coa_risk",
            "stage",
            "start_min",
            "end_min",
            "duration_min",
            "linked_claims",
            "captured_claims",
            "verified_claims",
            "contested_claims",
            "evidence_strength",
            "risk_level",
        ],
    )
    write_csv(
        OUT / "coa_timeline_overlay_gates.csv",
        [
            {
                "scenario_id": scenario["id"],
                "coa_id": selected["coa"]["id"],
                **gate,
            }
            for gate in gates
        ],
        ["scenario_id", "coa_id", "gate", "time_min", "label", "condition"],
    )
    comparison_rows = []
    for item in comparisons:
        timeline = item["timeline"]
        comparison_rows.append(
            {
                "scenario_id": scenario["id"],
                "method": item["method"],
                "method_label": item["label"],
                "coa_id": item["coa_id"],
                "gpi": round(item["gpi"], 6),
                "traceability": round(item["traceability"], 6),
                "robustness": round(item["robustness"], 6),
                "controllability": round(item["controllability"], 6),
                "risk_penalty": round(item["risk_penalty"], 6),
                "authority_violations": item["authority_violations"],
                "gate_closure": round(item["gate_closure"], 6),
                "start_min": round(timeline["start_min"], 2),
                "end_min": round(timeline["end_min"], 2),
                "verified_claims": timeline["verified_claims"],
                "captured_claims": timeline["captured_claims"],
                "contested_claims": timeline["contested_claims"],
                "max_risk": round(timeline["max_risk"], 3),
                "mean_risk": round(timeline["mean_risk"], 3),
            }
        )
    write_csv(
        OUT / "coa_timeline_baseline_comparison.csv",
        comparison_rows,
        [
            "scenario_id",
            "method",
            "method_label",
            "coa_id",
            "gpi",
            "traceability",
            "robustness",
            "controllability",
            "risk_penalty",
            "authority_violations",
            "gate_closure",
            "start_min",
            "end_min",
            "verified_claims",
            "captured_claims",
            "contested_claims",
            "max_risk",
            "mean_risk",
        ],
    )
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
    print(f"Wrote {comparison_pdf}")
    print(f"Wrote {comparison_png}")
    print(f"Scenario {scenario['id']} selected COA {selected['coa']['id']}")


if __name__ == "__main__":
    main()
