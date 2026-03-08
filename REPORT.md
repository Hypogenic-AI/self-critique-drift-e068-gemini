# REPORT

## 1. Executive Summary
This project tested whether self-critique changes internal representations (residual stream) in structured ways rather than only changing outputs.

Key finding: self-critique and revision conditions produced substantial activation drift, but on this setup they did not yield statistically significant accuracy gains over direct prompting; evidence for structured internal change is mixed and condition-dependent.

Practical implication: reflection prompts can strongly move internal states even when behavioral gains are small or inconsistent, so "improved answer quality" and "mechanistic change" should be evaluated separately.

## 2. Goal
### Hypothesis
Self-critique induces structured shifts in residual-stream representations, improving correctness separability and convergence toward a stable reasoning manifold.

### Why Important
Self-reflection prompting is widely used, but it is unclear whether gains reflect meaningful internal computation changes or shallow re-sampling.

### Problem Solved
This work provides an executable evaluation harness that jointly measures:
1. Behavioral performance
2. Representation drift
3. Probe separability/manifold structure
4. Multi-round trajectory stability

### Expected Impact
Supports more rigorous evaluation of recursive self-improvement pipelines and mechanistic-interpretability claims around reflection prompts.

## 3. Data Construction
### Dataset Description
- GSM8K test split (`datasets/gsm8k`): 1,319 math reasoning examples
- CommonsenseQA validation split (`datasets/commonsense_qa`): 1,221 multiple-choice reasoning examples
- TruthfulQA generation split was available but not used in the final mechanistic run due scope/time prioritization.

### Example Samples
| Dataset | Question (shortened) | Label |
|---|---|---|
| GSM8K | "Janet’s ducks lay 16 eggs per day..." | `18` |
| CommonsenseQA | "A revolving door ... security measure at a what?" | `A` |
| GSM8K | "Natalia sold clips..." | numeric final answer |

### Data Quality
- Missing values: 0% in selected splits
- Duplicate questions: 0 detected in selected splits
- GSM8K answer parse success (`####` extraction): 1319/1319
- CommonsenseQA class distribution (validation): `A:239, B:255, C:241, D:251, E:235`
- Question length (chars):
  - GSM8K mean 239.87, std 97.57
  - CommonsenseQA mean 68.10, std 28.32

### Preprocessing Steps
1. Loaded local HuggingFace datasets from `datasets/`.
2. Sampled deterministic subsets via fixed seed.
3. Converted CommonsenseQA choices into explicit prompt options.
4. Parsed labels:
   - GSM8K: numeric extraction from gold `####` answer
   - CommonsenseQA: answer letter (`A-E`)
5. Parsed model predictions from `FINAL: ...` line with regex fallbacks.

### Train/Val/Test Splits
No model training/fine-tuning was performed. Evaluation subsets:
- Main mechanistic run (Qwen2.5-1.5B): 25 GSM8K + 25 CommonsenseQA = 50
- Small-model run (Qwen2.5-0.5B): 12 GSM8K + 12 CommonsenseQA = 24
- OpenAI API behavioral run (gpt-4.1): first 20 items from sampled pool

## 4. Experiment Description
### Methodology
#### High-Level Approach
For each question, generate a direct draft answer, then apply revision variants (rewrite control, self-critique 1 round, self-critique 3 rounds, external critique). Capture residual-stream vectors from selected layers for each stage and compare geometry/behavior.

#### Why This Method
This directly operationalizes "cognitive change" as measurable movement/structure in hidden representations while controlling task and prompt context.

Alternatives considered but not implemented in this run:
- Full CCA/SVCCA suite across all tokens
- TruthfulQA mechanistic analysis
- Larger sample sizes per condition/model

### Implementation Details
#### Tools and Libraries
- Python 3.12.8
- torch 2.10.0+cu128
- transformers 5.3.0
- datasets 4.4.0
- numpy 2.3.5
- scipy 1.17.1
- scikit-learn 1.8.0
- statsmodels 0.14.6
- openai 2.26.0

#### Algorithms/Models
- Local mechanistic models:
  - `Qwen/Qwen2.5-1.5B-Instruct` (main)
  - `Qwen/Qwen2.5-0.5B-Instruct` (size comparison)
- API behavioral model:
  - `gpt-4.1`
- Probing:
  - Logistic regression over layer vectors for correctness classification

#### Hyperparameters
| Parameter | Value | Selection Method |
|---|---|---|
| seed | 42 | fixed reproducibility |
| max_new_tokens | 48 (main run) | runtime/quality tradeoff |
| temperature | 0.0 | deterministic control |
| selected layers | [1, 7, 14, 21, 28] for 1.5B; analogous quartiles for 0.5B | coverage of depth |
| probe split | 70/30 | standard holdout |
| bootstrap resamples | 1000 | stable CI estimate |
| multiple testing | Benjamini-Hochberg FDR | planned correction |

#### Training/Analysis Pipeline
1. Generate direct draft.
2. Generate revision variants under controlled prompts.
3. Forward pass each text with `output_hidden_states=True`.
4. Extract final-token residual vectors from selected layers.
5. Compute:
   - accuracy
   - cosine drift (pre→post)
   - CKA (pre vs post matrices)
   - probe AUC/F1
   - silhouette (PCA-2D)
   - multi-round step drift
6. Run paired significance tests and FDR correction.
7. Save JSON, raw records, and plots.

### Experimental Protocol
#### Reproducibility Information
- Number of runs for primary results: 1 per model condition set
- Random seed: 42
- Hardware:
  - GPU: 2x NVIDIA GeForce RTX 3090 (24GB each)
  - CUDA available: yes
- Mixed precision: bfloat16 inference on local models
- Effective batch size: 1 sequence (activation extraction per-sample)
- Runtime (observed):
  - 1.5B run (50 samples): ~8-10 minutes including loading
  - 0.5B run (24 samples): ~3-5 minutes

#### Evaluation Metrics
- Accuracy: exact correctness for GSM8K numeric/CSQA letter answers
- Cosine drift: 1-cosine similarity between draft and revised layer vectors
- CKA: similarity of representation geometry pre vs post
- Probe AUC/F1: separability of correct vs incorrect from activations
- Silhouette: class separation in reduced manifold
- Step drift (round1/2/3): trajectory stability over critique rounds

### Raw Results
#### Accuracy Table
| Model | Direct | Rewrite | Self-Critique 1 | Self-Critique 3 | External Critique |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B (n=50) | 0.26 | 0.30 | 0.28 | 0.22 | 0.34 |
| Qwen2.5-0.5B (n=24) | 0.042 | 0.167 | 0.083 | 0.042 | 0.25 |

#### Statistical Comparison (1.5B, vs Direct)
| Comparison | Test | p | q (FDR) | Effect |
|---|---|---:|---:|---:|
| Rewrite vs Direct | Wilcoxon | 0.530 | 0.802 | 0.0 |
| Self1 vs Direct | Wilcoxon | 0.802 | 0.802 | 0.0 |
| Self3 vs Direct | Wilcoxon | 0.618 | 0.802 | 0.0 |
| External vs Direct | Wilcoxon | 0.359 | 0.802 | 0.0 |

No condition achieved statistical significance at `alpha=0.05` after correction.

#### Layer-Wise Drift (1.5B, last selected layer=28)
| Condition | Mean Cosine Drift | CKA (pre vs post) |
|---|---:|---:|
| Rewrite control | 0.3999 | 0.2354 |
| Self-critique 1 | 0.3683 | 0.2034 |
| Self-critique 3 | 0.4230 | 0.1890 |
| External critique | 0.3439 | 0.2445 |

#### Multi-Round Stability (1.5B, layer 28)
Step drift decreases with rounds:
- round1: 0.3683
- round2: 0.3598
- round3: 0.3347

This is consistent with partial trajectory convergence.

#### Probe AUC (1.5B, layer 28)
Pre→post change:
- Rewrite: 0.818 → 0.545
- Self-critique 1: 0.818 → 0.750
- Self-critique 3: 0.818 → 0.889
- External critique: 0.818 → 0.300

Interpretation: only multi-round self-critique improved correctness separability at this layer in this run.

#### API Behavioral Triangulation (gpt-4.1, n=20)
- Direct accuracy: 1.00
- Self-critique-1 accuracy: 0.95
- Wilcoxon p=1.0 (no significant difference)

### Output Locations
- Combined metrics: `results/metrics_combined.json`
- Main model metrics: `results/metrics_Qwen_Qwen2.5-1.5B-Instruct.json`
- Small model metrics: `results/metrics_Qwen_Qwen2.5-0.5B-Instruct.json`
- Raw per-example outcomes: `results/raw_*.jsonl`
- Plots: `results/plots/`

## 5. Result Analysis
### Key Findings
1. Reflection/revision prompts caused substantial representation drift (cosine drift ~0.34-0.42 at deep layer in 1.5B).
2. Behavioral gains over direct prompting were small and not statistically significant on the main run.
3. Multi-round self-critique showed decreasing step drift, suggesting trajectory stabilization.
4. Correctness separability improved for self-critique-3 at the deepest layer (AUC increase), but not consistently across all conditions.

### Hypothesis Testing Results
- H1 (drift magnitude): supported (clear non-trivial pre/post drift)
- H2 (structured drift): partially supported (self-critique-3 improved probe separability; other conditions did not)
- H3 (convergence): supported in trajectory metric (decreasing step drift)
- H4 (specificity): mixed; external critique had best raw accuracy on these subsets but weaker separability geometry

### Comparison to Baselines
Compared to direct prompting (1.5B), highest raw improvement was external critique (+0.08 absolute), but non-significant. Self-critique-3 underperformed direct on accuracy while showing stronger geometric shift.

### Visualizations
Generated plots in `results/plots/`:
- `accuracy_*.png`
- `drift_layers_*.png`
- `trajectory_*.png`

Each plot includes title, labeled axes, and legends.

### Surprises and Insights
- Large drift can occur without strong behavioral gains.
- Multi-round reflection can stabilize trajectories while not improving final accuracy.
- External critique template was competitive behaviorally in this low-sample setup.

### Error Analysis
- GSM8K remained difficult for both open models; most errors are arithmetic/step propagation.
- CommonsenseQA had higher baseline correctness and more condition sensitivity.
- Small-model run exhibited severe class imbalance for probe labels, yielding unstable AUC/F1.

### Limitations
- Sample sizes were modest (especially 0.5B and API run).
- Probe estimates for small model are noisy due low positive-class count.
- Hidden-state extraction used final token only; richer token-trajectory methods may capture more structure.
- No inter-model activation alignment (e.g., SVCCA across models) in this run.
- API run measured behavior only (no internals).

## 6. Conclusions
Self-critique does induce measurable internal representational drift in residual-stream states in this setup. Evidence for "meaningful" restructuring is partial: multi-round self-critique showed signs of improved correctness separability and trajectory stabilization, but behavioral improvements were not statistically significant on the main sample.

The central answer is therefore nuanced: reflection appears to reshape internals, but reliable performance gains and robust structure-performance coupling need larger controlled studies.

### Confidence in Findings
Moderate confidence in the presence of drift/convergence signals; low-to-moderate confidence in claims about accuracy improvements due to sample size and variability.

## 7. Next Steps
### Immediate Follow-ups
1. Increase sample sizes (>=200 per condition) and include TruthfulQA scoring to reduce variance.
2. Add stronger controls (equal-token no-critique, shuffled critique text) to isolate critique semantics.
3. Compute token-level trajectory metrics and SVCCA/Procrustes across all layers.

### Alternative Approaches
- Use attention-pattern drift and MLP neuron activation sparsity as complementary mechanistic signals.
- Use verifier-based correctness labels (process-level) rather than final-answer-only labels.

### Broader Extensions
- Compare self-critique vs human-written critique at matched length/quality.
- Evaluate additional model families and larger API models with repeated trials.

### Open Questions
- When does representational drift become predictive of reliable accuracy gain?
- Which layers contain the most causally relevant reflection-induced changes?
- Is convergence associated with correctness or merely with stylistic stabilization?

## References
- Shinn et al., Reflexion (2023)
- Madaan et al., Self-Refine (2023)
- Zelikman et al., Quiet-STaR (2024)
- Venhoff et al., Reasoning via Steering Vectors (2025)
- Cobbe et al., Training Verifiers (2021)
- Lin et al., TruthfulQA (2021)
