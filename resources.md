## Resources Catalog: Representation Drift Under Self-Reflection

### Summary
This document catalogs all gathered resources, including 8 key papers, 3 code repositories for representation analysis, and 4 datasets for evaluating reasoning and self-awareness.

### Papers
Total papers downloaded: 8 (Stored in `papers/`)

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| From Emergence to Control | Zhu et al. | 2025 | papers/2506.12217_From_Emergence_to_Control.pdf | Self-reflection vectors (Diff-of-means) |
| ReflCtrl | Yan et al. | 2025 | papers/2512.13979_ReflCtrl.pdf | Representation Engineering (RepE) for reflection |
| Factual Self-Awareness | Tamoyan et al. | 2025 | papers/2505.21399_Factual_Self_Awareness.pdf | Linear features in residual stream for recall |
| Self-Generated Recognition | Ackerman et al. | 2024 | papers/2410.02064_Self_Generated_Recognition.pdf | Self-authorship vector in residual stream |
| Metacognitive Monitoring | Li et al. | 2025 | papers/2505.13763_Metacognitive_Monitoring.pdf | Neurofeedback for internal state reporting |
| Instruct-of-Reflection | Liu et al. | 2025 | papers/2503.00902_Instruct_of_Reflection.pdf | Iterative reflection capabilities |
| Brain Rot! | Xing et al. | 2025 | papers/2510.13928_Brain_Rot.pdf | Measuring representational drift |
| Learning from All | Yang et al. | 2025 | papers/2510.04142_Learning_from_All.pdf | Concept alignment and drifting dynamics |

### Datasets
Total datasets downloaded: 4 (Stored in `datasets/`)

| Name | Source | Task | Location | Notes |
|------|--------|------|----------|-------|
| GSM8K | HuggingFace | Math Reasoning | datasets/gsm8k/ | Standard benchmark for reasoning |
| StrategyQA | HuggingFace | Multi-step QA | datasets/strategy_qa/ | Good for complex deliberation |
| ARC Challenge | HuggingFace | Science QA | datasets/arc_challenge/ | Challenging reasoning task |
| Factual Recall | GitHub (UKPLab) | Truthfulness | code/self_awareness/self_aware/data/ | Entities/Relations/Attributes |

### Code Repositories
Total repositories cloned: 3 (Stored in `code/`)

| Name | URL | Purpose | Location | Key Tools |
|------|-----|---------|----------|-----------|
| ProbingReflection | xzAscC/ProbingReflection | Vector extraction | code/probing_reflection/ | scripts/extract_reflection_vector.py |
| ReflCtrl | Trustworthy-ML-Lab/ReflCtrl | Steering/RepE | code/refl_ctrl/ | collect_activation.py, extract_dir.py |
| Self-Awareness | UKPLab/arxiv2025-self-awareness | Linear Probing | code/self_awareness/ | Probing factual recall states |

### Recommendations for Experiment Design

1. **Primary Dataset**: **GSM8K** (Main) and **Factual Recall** (to test if reflection improves internal conviction).
2. **Models**: Recommend using **Qwen2.5-1.5B** or **Llama-3-8B** as they are the focus of these papers.
3. **Metric**: **Linear Separability** of internal states before/after reflection, **Cosine Similarity** (drift) between steps.
4. **Method**: Adapt **Difference-of-Means** from `probing_reflection` and **Stepwise Steering** from `refl_ctrl` to analyze how the "reflection vector" shifts over multiple critique iterations.

## Experiment Execution Log (2026-03-08)
- Implemented end-to-end experiment runner: `src/run_representation_drift.py`.
- Ran mechanistic experiments on:
  - `Qwen/Qwen2.5-1.5B-Instruct` (n=50)
  - `Qwen/Qwen2.5-0.5B-Instruct` (n=24)
- Ran behavioral API triangulation on `gpt-4.1` (n=20, 60 API calls).
- Saved outputs to `results/metrics_combined.json`, `results/raw_*.jsonl`, and `results/plots/`.
- Added final documentation: `REPORT.md`, `README.md`, `CODE_WALKTHROUGH.md`.
