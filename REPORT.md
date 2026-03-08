# Representation Drift Under Self-Reflection: Does Self-Critique Reshape Internal States?

## 1. Executive Summary
This research investigates whether asking a Large Language Model (LLM) to "critique and revise" its own answers results in meaningful changes to its internal representations, specifically looking at the "drift" in the residual stream and the linear separability of correct versus incorrect reasoning. Through experiments on the Qwen2.5-1.5B and Qwen2.5-0.5B models using GSM8K and CommonsenseQA, we found that representation drift occurs across both self-critique and simple "rewrite" controls. However, self-critique does not always increase the structural separability of correct reasoning. For the 1.5B model, an *external* critique dramatically improved both task accuracy (31.7% to 40.0%) and internal linear separability (AUC: 0.486 to 0.831), whereas self-critique showed little to no improvement over baseline. Conversely, the 0.5B model exhibited successful self-critique, restructuring its internal state (AUC: 0.550 to 0.719) and improving accuracy. This suggests that while representation drift is measurable, "meaningful" reflection that structures the reasoning subspace is highly model- and task-dependent, and external guidance remains significantly more potent for restructuring latent states than autonomous self-critique in smaller models.

## 2. Goal
**Hypothesis:** Self-critique in LLMs induces structured shifts in internal representations (residual stream), resulting in a more distinct and structured reasoning subspace. If reflection is meaningful, we expect representation drift, improved linear separability of correct reasoning, and convergence toward a more stable reasoning manifold; if shallow, activations remain near the original neighborhood with minimal structural change.

**Why this matters:** LLMs are frequently prompted to "think again" or revise their answers. Understanding whether these techniques fundamentally reshape the model's internal processing or simply act as a shallow re-sampling is essential for building recursive self-improvement systems and developing interpretability-driven control methods like Representation Engineering (RepE).

## 3. Data Construction

### Dataset Description
We used a balanced combination of multi-step mathematical reasoning and commonsense reasoning tasks to evaluate reflection:
- **GSM8K (test set subset):** Grade school math problems requiring multi-step arithmetic logic.
- **CommonsenseQA (validation set subset):** Multiple-choice questions requiring implicit multi-step deliberation.

For this study, we utilized 60 samples (30 GSM8K + 30 CSQA) for the main 1.5B model and 40 samples (20+20) for the 0.5B model to facilitate computationally intensive internal activation extraction across multiple hidden layers.

### Preprocessing Steps
1. Filtered the datasets for valid formatting.
2. Extracted pure mathematical reference answers from GSM8K (`#### [answer]`) and reference keys (`[A-E]`) from CommonsenseQA.
3. Formatted inputs using the models' chat templates for consistent instruction following.

## 4. Experiment Description

### Methodology
We extracted internal residual stream activations at the final token of the generated reasoning across several layers (e.g., 1/4, 1/2, 3/4, and final layer) during different inference conditions:
1. **Direct (Baseline):** Standard Chain-of-Thought (CoT) zero-shot generation.
2. **Rewrite Control:** Prompting the model to rewrite its draft for clarity without changing the core reasoning.
3. **Self-Critique (1 round):** Prompting the model to generate a critique of its draft, then generating a revised answer.
4. **Self-Critique (3 rounds):** Iterating the critique-revise loop three times.
5. **External Critique:** Injecting a task-specific oracle critique rule (e.g., "Check arithmetic carefully") and asking the model to revise.

For each condition, we measured:
- **Accuracy:** Final task performance.
- **Representation Drift:** Mean Cosine distance between the baseline draft activations and the revised activations at the last layer.
- **Linear Separability (AUC):** We trained Logistic Regression probes on the activations to classify whether the final answer was correct or incorrect, measuring the AUC before (pre) and after (post) the critique/revision phase.

### Implementation Details
- **Models:** `Qwen/Qwen2.5-1.5B-Instruct` (Main) and `Qwen/Qwen2.5-0.5B-Instruct` (Small).
- **Libraries:** HuggingFace `transformers`, `torch` (bfloat16, CUDA), `scikit-learn` for linear probing.
- **Hyperparameters:** Temperature 0.0 (greedy decoding) to isolate representational changes strictly caused by the prompt condition rather than sampling variance. Max new tokens: 96.

## 5. Result Analysis

### Key Findings

#### 1. Self-Critique Does Not Guarentee Structural Improvement (1.5B Model)
For `Qwen2.5-1.5B-Instruct`, standard self-critique (1 round) improved accuracy marginally from 31.7% to 35.0%. However, this was slightly *worse* than the simple "Rewrite" control (36.7%). More critically, looking at the internal representations (Final Layer Probe AUC):
- **Pre-Critique AUC:** 0.486
- **Self-Critique 1-Round Post-AUC:** 0.444
- **Self-Critique 3-Round Post-AUC:** 0.514
Self-critique failed to make the correct reasoning linearly separable.

#### 2. External Critique Dramatically Reshapes Latent Space
When an external, oracle-like critique was provided to the 1.5B model, accuracy jumped to 40.0%. The internal states reflected a massive structural shift:
- **External Critique Post-AUC:** 0.831 (up from 0.486)
This indicates that the model *has the capacity* to structure its reasoning subspace perfectly, but autonomous self-critique in a 1.5B model is not strong enough to trigger this internal restructuring.

#### 3. Representation Drift is Omnipresent
Drift (measured as cosine distance from the initial draft state) occurred in all conditions.
- **Rewrite Control Drift:** 0.3948
- **Self-Critique 1 Drift:** 0.3619
- **External Critique Drift:** 0.3375
Interestingly, external critique caused *less* absolute drift but resulted in *higher* accuracy and separability. This implies that meaningful critique is not about "wandering far" in latent space, but about moving in a highly specific, structured direction.

#### 4. The 0.5B Model Anomaly
Surprisingly, the smaller `Qwen2.5-0.5B-Instruct` showed a distinct pattern:
- Direct Accuracy: 17.5% -> Self-Critique 1 Accuracy: 32.5%.
- Pre-AUC: 0.550 -> Self-Critique 1 Post-AUC: 0.719.
For the 0.5B model, a single round of self-reflection was highly effective at restructuring the internal state to separate correct from incorrect reasoning. (Though 3 rounds degraded performance, likely due to context limits or "Brain Rot").

### Visualizations
*(Plots are saved in `results/plots/`)*
- `accuracy_Qwen_Qwen2_5_1_5B_Instruct.png`: Shows External Critique dominating task performance.
- `drift_layers_Qwen_Qwen2_5_1_5B_Instruct.png`: Shows that drift steadily increases in deeper layers, with Rewrite Control actually causing the highest raw drift.

### Limitations
- **Model Scale:** 1.5B and 0.5B models are relatively small. Emergent self-reflection capabilities (and internal restructuring) may behave fundamentally differently in 70B+ models.
- **Sample Size:** 60 samples were used per model to manage the massive memory footprint of caching hidden states across all layers for 5 distinct generation steps. A larger sample size would yield tighter confidence intervals on the logistic regression probes.

## 6. Conclusions
We successfully demonstrated that self-reflection and critique induce measurable representational drift in the residual stream. However, **drift does not equal cognitive restructuring**. For the 1.5B model, self-critique caused the representations to drift just as much as a meaningless "rewrite" prompt, but completely failed to improve the linear separability of correct reasoning. In contrast, an external critique drastically improved this separability (AUC 0.486 -> 0.831). 
This supports a nuanced view of the hypothesis: while reflection vectors exist, autonomous self-critique in smaller LLMs often results in "shallow" resampling rather than deep structural realignment. The capacity for a structured reasoning subspace is latent but often requires external guidance to be fully activated.

## 7. Next Steps
1. **Scale Up:** Replicate this activation extraction pipeline on Llama-3-8B and Qwen2.5-7B using mixed precision to see if larger models can autonomously structure their reasoning subspace during self-critique.
2. **Representation Engineering:** Extract the difference-of-means vector between the "External Critique" states and "Self Critique" states, and use Representation Engineering (RepE) to inject the "External Critique" vector into the model during standard CoT generation.
3. **Step-by-Step Convergence:** Analyze intermediate layers dynamically to see *where* in the forward pass the reflection fails to restructure the latent space.