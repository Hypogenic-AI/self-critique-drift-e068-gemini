## Literature Review: Representation Drift Under Self-Reflection

### Research Area Overview
Large Language Models (LLMs) exhibit emergent behaviors such as self-reflection—the ability to evaluate and revise their own reasoning. Recent research explores these behaviors through the lens of internal representations, specifically looking at "reflection vectors" and "self-awareness signals" in the residual stream. This research area aims to understand if self-critique is a "shallow" format change or a "deep" reshaping of internal states toward a more structured reasoning subspace.

### Key Papers

#### 1. From Emergence to Control: Probing and Modulating Self-Reflection in Language Models (Zhu et al., 2025)
- **Key Contribution**: Demonstrates that self-reflection is a latent ability even in pretrained models. Constructs a "self-reflection vector" in activation space.
- **Methodology**: Uses "Reflection-Inducing Probing" (injecting reflection traces) and identifies a "difference-of-means" vector between reflective and non-reflective hidden states.
- **Relevance**: Directly supports the hypothesis that reflection has a distinct internal signature that can be modulated.

#### 2. ReflCtrl: Controlling LLM Reflection via Representation Engineering (Yan et al., 2025)
- **Key Contribution**: Uses Representation Engineering (RepE) to steer reflection frequency and cost.
- **Methodology**: Identifies a "reflection direction" in the latent space (MLP and attention outputs) and uses it for stepwise steering.
- **Relevance**: Connects internal uncertainty to reflection and shows that reflection behavior is steerable via internal state manipulation.

#### 3. Factual Self-Awareness in Language Models: Representation, Robustness, and Scaling (Tamoyan et al., 2025)
- **Key Contribution**: Identifies linear features in the residual stream that dictate whether a model will correctly recall a fact.
- **Methodology**: Trains linear probes on final token residuals using a custom factual recall dataset.
- **Relevance**: Provides a framework for measuring "internal conviction" or "self-awareness" before generation, which is likely affected by self-critique.

#### 4. Inspection and Control of Self-Generated-Text Recognition Ability (Ackerman & Panickssery, 2024)
- **Key Contribution**: Finds a "self-authorship" vector in the residual stream.
- **Relevance**: Suggests that models can internally distinguish their own generated reasoning, a prerequisite for meaningful self-critique.

#### 5. LLMs Can Get "Brain Rot"! (Xing et al., 2025)
- **Key Contribution**: Analyzes representational drift caused by low-quality data.
- **Relevance**: Provides methodology for measuring "drift" in internal states over time or iterations.

### Common Methodologies
1. **Difference-of-Means Probing**: Subtracting mean activations of "negative" samples from "positive" samples to find a direction (Zhu et al., Yan et al.).
2. **Linear Probing**: Training simple classifiers on residual stream activations to detect latent properties like truthfulness or self-awareness (Tamoyan et al.).
3. **Representation Engineering (RepE)**: Directly manipulating activations along identified directions to steer behavior (Yan et al.).
4. **Activation Clustering (UMAP/t-SNE)**: Visualizing the separation of internal states in high-dimensional space.

### Standard Baselines
- **Vanilla CoT**: Standard chain-of-thought without explicit self-critique.
- **Zero-shot Reflection**: Prompting for reflection without internal state modulation.
- **Random Direction Steering**: Using a random vector as a control for intervention experiments.

### Recommended Datasets
- **GSM8K/MATH**: For complex reasoning and verifiable correctness.
- **StrategyQA**: For multi-step reasoning.
- **ARC (Challenge)**: For challenging scientific reasoning.
- **Factual Recall**: For measuring internal conviction.

### Gaps and Opportunities
While previous work has identified "reflection directions", it remains unclear how these directions *evolve* or *drift* during multiple rounds of self-critique. Does the reasoning subspace become more "linearly separable" as a model iterates? This is the core of our proposed research.
