# C2 LLM-Agent Governance Simulator

This repository is the public reproduction package for the manuscript
"A Computational Governance Framework for LLM-Enabled C2 Organizational Cognitive Collaboration in the Military Decision-Making Process".

The package contains a shareable symbolic simulator, synthetic scenario generator, experiment scripts, CSV outputs, and manuscript figures. It does not contain classified operational data, live LLM calls, human-subject data, or redistributed third-party raw datasets.

## Contents

- `code/c2_experiments.py`: synthetic scenario generator, policy simulator, comparison experiments, ablations, sensitivity analysis, statistical tests, and aggregate figures.
- `code/run_l9_calibration.py`: L9 orthogonal calibration and validation script.
- `code/generate_coa_timeline_overlay.py`: data-driven COA execution timeline figures for the selected case study.
- `outputs/scenario_records.csv`, `outputs/scenario_claims.csv`, `outputs/candidate_coas.csv`: generated scenario-, claim-, and candidate-level records for the released test set.
- `outputs/experiment_*.csv`: policy-level and aggregate outputs used by the manuscript tables, diagnostics, and statistical tests.
- `outputs/l9_*.csv`: orthogonal calibration, range analysis, and validation outputs.
- `outputs/coa_timeline_*.csv`: data records behind the COA execution timeline figures.
- `figures/*.pdf`: generated experimental figures used in the manuscript.
- `docs/SIMULATOR_SPECIFICATION.md`: implemented simulator rules, parameter definitions, scenario distributions, and metric equations.
- `config/calibration_config.json`: calibration seeds, L9 factor levels, and selected default parameters.
- `experiment_config.json`: seeds, scenario counts, selected parameters, and reproduction commands.
- `requirements.txt`: Python dependencies.

## Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main experiments:

```bash
python code/c2_experiments.py
```

Run the L9 calibration and validation:

```bash
python code/run_l9_calibration.py
```

Run the COA timeline figure script:

```bash
python code/generate_coa_timeline_overlay.py
```

The main script regenerates the scenario/candidate/claim CSV files, the `outputs/experiment_*.csv` files, and the aggregate experimental figures. The calibration script regenerates `outputs/l9_calibration_results.csv`, `outputs/l9_range_analysis.csv`, and `outputs/l9_validation_results.csv`. All scripts write to the repository-level `outputs/` and `figures/` directories.
