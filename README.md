# Representation Drift Under Self-Reflection

This project tests whether self-critique changes LLM internal states (residual stream) in structured ways, beyond output-only improvements. It runs local mechanistic experiments on open models and a small behavioral API comparison.

## Key Findings
- Reflection/revision prompts caused substantial layer-wise representation drift.
- On the 1.5B main run (n=50), accuracy differences vs direct prompting were not statistically significant.
- Multi-round self-critique showed decreasing step drift (trajectory stabilization).
- Deep-layer probe separability improved for multi-round self-critique in the main model.

## Reproduce
```bash
source .venv/bin/activate
python src/run_representation_drift.py --n-gsm 25 --n-csqa 25 --n-gsm-small 12 --n-csqa-small 12 --max-new-tokens 48
```

## Outputs
- Main report: `REPORT.md`
- Plan: `planning.md`
- Script: `src/run_representation_drift.py`
- Metrics: `results/metrics_combined.json`
- Plots: `results/plots/`

## File Structure
- `datasets/`: local benchmark data
- `code/`: cloned baseline repos
- `src/`: experiment code
- `results/`: metrics, raw outputs, visualizations
