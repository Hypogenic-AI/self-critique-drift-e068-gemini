# Outline: Self-Critique Representation Drift Paper

## Title
- Reflection Without Guaranteed Gains: Representation Drift and Trajectory Stabilization Under Self-Critique in Small LLMs

## Abstract
- Problem: unclear if self-critique changes internal computation vs output resampling
- Approach: paired behavioral + residual-stream geometry analysis across revision conditions
- Key numbers: drift 0.34-0.42; no significant accuracy gain (all q=0.802); step drift decreases 0.368->0.335
- Significance: internal change and output gain should be evaluated separately

## Introduction
- Hook: self-critique is deployed widely but mechanism unclear
- Gap: prior reflection-vector work is mostly static; little dynamic trajectory analysis across rounds
- Approach: executable harness measuring behavior, drift, CKA, probe separability, and trajectory stability
- Result preview: strong drift, mixed structure signals, non-significant behavioral gains
- Contributions: benchmark protocol; empirical findings; triangulation across model sizes and API model

## Related Work
- Self-reflection prompting methods: Reflexion, Self-Refine, Quiet-STaR
- Representation-level control/probing: Zhu et al., Yan et al., Tamoyan et al., Ackerman & Panickssery
- Drift and reliability: Xing et al.; verifier/evaluation work (Cobbe et al., TruthfulQA)
- Positioning: dynamic multi-round residual trajectory analysis with controls

## Methodology
- Tasks and datasets: GSM8K + CommonsenseQA subsets
- Conditions: direct, rewrite, self-critique-1, self-critique-3, external critique
- Models: Qwen2.5-1.5B, Qwen2.5-0.5B, GPT-4.1 behavioral triangulation
- Metrics: accuracy, cosine drift, CKA, probe AUC/F1, silhouette, round-to-round drift
- Stats: paired Wilcoxon + BH-FDR

## Results
- Table 1: accuracy across conditions/models
- Table 2: significance tests vs direct
- Table 3: layer-28 geometry and probe changes
- Figure 1: main model accuracy plot
- Figure 2: layer-wise drift plot
- Figure 3: trajectory convergence plot
- Ablation/sensitivity: model size comparison and condition effects

## Discussion
- Interpretation: drift != guaranteed performance gain
- Mechanistic implications: partial evidence for structured change in self-critique-3
- Limitations: sample size, final-token only, single run, noisy small-model probes
- Broader implications and evaluation recommendations

## Conclusion
- Summarize contributions and findings
- Key takeaway: separate internal-change metrics from task gains
- Future work: larger controlled runs, token-level analyses, SVCCA/Procrustes, expanded datasets

## Evidence Mapping
- All core numeric claims from REPORT tables/metrics listed in user prompt
- Plots sourced from results/plots/*.png
