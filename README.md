# C2 LLM-Agent Governance Simulator

This repository is the public reproduction package for the manuscript
"A Computational Governance Framework for LLM-Enabled C2 Organizational Cognitive Collaboration in the Military Decision-Making Process".

The package contains a shareable symbolic simulator, synthetic scenario generator, experiment scripts, CSV outputs, and manuscript figures. It does not contain classified operational data, live LLM calls, human-subject data, or redistributed third-party raw datasets.

## Contents

- `code/c2_experiments.py`: synthetic scenario generator, policy simulator, comparison experiments, ablations, sensitivity analysis, statistical tests, and aggregate figures.
- `code/generate_coa_timeline_overlay.py`: data-driven COA execution timeline figures for the selected case study.
- `outputs/experiment_*.csv`: raw and aggregate outputs used by the manuscript tables, diagnostics, and statistical tests.
- `outputs/coa_timeline_*.csv`: data records behind the COA execution timeline figures.
- `figures/*.pdf`: generated experimental figures used in the manuscript.
- `docs/SIMULATOR_SPECIFICATION.md`: implemented simulator rules, parameter definitions, scenario distributions, and metric equations.
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

Run the COA timeline figure script:

```bash
python code/generate_coa_timeline_overlay.py
```

The main script regenerates the `outputs/experiment_*.csv` files and the aggregate experimental figures.

## Manuscript Link

After uploading this folder to an anonymous or public GitHub repository, replace the manuscript placeholder URL with the real repository URL:

```text
https://github.com/USERNAME/c2-llm-agent-governance-simulator
```
# c2-llm-agent-governance-simulator
