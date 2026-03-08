# Code Walkthrough

## Code Structure Overview
- `src/run_representation_drift.py`: complete pipeline (data load, prompting conditions, activation extraction, statistics, plots, API triangulation).

## Environment Setup
```python
# key libraries used in script
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
```

## Key Functions
### `load_samples(data_root, n_gsm, n_csqa, seed)`
Purpose: deterministic subset construction for GSM8K + CommonsenseQA.

### `build_prompt(task, style, question_text, draft="", critique="")`
Purpose: prompt templates for direct, rewrite control, self-critique, and revise steps.

### `get_layer_vectors(model, tokenizer, text, selected_layers)`
Purpose: extracts final-token residual vectors from selected layers.

### `run_local_model_experiment(...)`
Purpose: executes all conditions, computes metrics, significance tests, probes, and plot outputs.

### `run_openai_behavioral(samples, results_dir, max_items=20)`
Purpose: behavioral-only API comparison on real OpenAI model.

## Data Pipeline
Raw dataset -> sampled items -> prompt variants -> model responses -> parsed predictions -> hidden state extraction -> metrics/statistics -> JSON/plots.

## How to Run
```bash
source .venv/bin/activate
python src/run_representation_drift.py --n-gsm 25 --n-csqa 25 --n-gsm-small 12 --n-csqa-small 12 --max-new-tokens 48
```

## Expected Runtime
- GPU (2x3090 available): ~12-18 minutes total for the above config.
- CPU-only: significantly slower.

## Reproducibility
- Seed fixed in script (`--seed`, default 42)
- Config saved to `results/config.json`
- Metrics saved to `results/metrics_*.json`
