# Representation Drift Under Self-Reflection

## Overview
This repository contains the code, data, and findings for the research project: "Representation Drift Under Self-Reflection: Does Self-Critique Reshape Internal States?" We investigated whether prompting an LLM to self-critique its own answers induces structured shifts in its internal residual stream representations, or if it merely acts as a shallow prompt re-sampling.

## Key Findings
- **Drift is omnipresent:** Asking a model to self-critique or simply "rewrite" an answer causes measurable representation drift in the residual stream (cosine distances ~0.35-0.4).
- **Self-Critique does not guarantee structural improvement:** For Qwen2.5-1.5B, self-critique did not improve the linear separability (AUC) of correct vs. incorrect answers compared to the initial draft.
- **External Critique is highly effective:** Injecting an external, oracle-like critique drastically restructured the internal latent space (Probe AUC improved from 0.486 to 0.831) and improved accuracy (31.7% -> 40.0%), suggesting the model has the *capacity* to structure its reasoning, but autonomous critique fails to reliably trigger it in the 1.5B model.
- **Scale dynamics:** The smaller Qwen2.5-0.5B model actually *did* show internal restructuring under autonomous self-critique, indicating that self-reflection dynamics are highly sensitive to model scale and capacity.

## How to Reproduce
1. **Environment Setup:** Ensure you have Python 3.10+ and a CUDA-capable GPU (at least 16GB VRAM recommended for the 1.5B model).
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install torch transformers accelerate datasets scikit-learn numpy matplotlib tqdm
   ```

2. **Run the Experiment:**
   Execute the experimental pipeline which automatically extracts hidden states and trains linear probes:
   ```bash
   python src/run_representation_drift.py --n-gsm 30 --n-csqa 30 --skip-openai
   ```
   This will output metrics, raw predictions, and plots to the `results/` directory.

## File Structure
- `src/run_representation_drift.py`: Main execution script that handles model inference, hidden state extraction, linear probing, and visualization.
- `datasets/`: Pre-downloaded subsets of GSM8K and CommonsenseQA used in the evaluation.
- `results/`: Contains the JSON metrics output, configuration logs, and generated plots.
- `REPORT.md`: Comprehensive final research report detailing methodology, full findings, limitations, and future work.
- `planning.md`: The original experimental design and hypothesis decomposition.

## Full Report
Please see [REPORT.md](./REPORT.md) for the complete research methodology, experimental protocol, and detailed analysis.