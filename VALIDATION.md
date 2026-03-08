# Validation Checklist

## Code Validation
- [x] Main pipeline script runs end-to-end (`src/run_representation_drift.py`).
- [x] Deterministic reproducibility smoke test run twice with same seed (`results/repro_run1`, `results/repro_run2`) produced identical accuracy summaries.
- [x] Outputs written to documented locations under `results/`.
- [x] Random seeds set in code.
- [x] No hardcoded absolute paths in runner.

## Scientific Validation
- [x] Paired tests used for condition comparisons (Wilcoxon/paired t depending normality).
- [x] Multiple-comparison correction (Benjamini-Hochberg FDR) applied.
- [x] Confidence intervals reported via bootstrap.
- [x] Limitations and alternative explanations documented in `REPORT.md`.
- [x] Confounds partially controlled (temperature=0, rewrite control, external critique condition).

## Documentation Validation
- [x] `REPORT.md` contains methodology, results, analysis, and limitations.
- [x] `README.md` includes overview and reproduction commands.
- [x] Plots include titles/axes/legends.
- [x] Code walkthrough provided (`CODE_WALKTHROUGH.md`).

## Output Validation
- [x] Metrics JSON generated (`results/metrics_combined.json`).
- [x] Raw records generated (`results/raw_*.jsonl`).
- [x] Visualization files generated (`results/plots/*.png`).
- [x] Config/environment snapshot generated (`results/config.json`).
