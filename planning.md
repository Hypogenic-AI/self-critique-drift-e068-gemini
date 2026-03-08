## Motivation & Novelty Assessment

### Why This Research Matters
LLMs are increasingly prompted to "think again" or critique themselves to improve reasoning performance. However, it's not well understood whether these performance gains correspond to a fundamental reshaping of the model's internal representations (e.g., toward a more coherent "reasoning subspace") or if it merely amounts to shallow resampling of the output distribution. Understanding this distinction is crucial for developing reliable, self-improving AI systems and informing the design of better prompting or representation engineering techniques.

### Gap in Existing Work
Existing work (like Zhu et al., Yan et al.) has identified specific "reflection vectors" or "self-awareness signals" in the residual stream that correlate with a model's propensity to self-correct. However, what is missing is a dynamic analysis of how these representations *drift* or evolve across multiple rounds of critique. Specifically, does the internal state converge toward a more stable, linearly separable manifold of "correct reasoning", or does it simply wander within its original neighborhood?

### Our Novel Contribution
We propose to measure the representation drift of internal states across multiple iterations of self-critique. By tracking the cosine similarity and linear separability of residual stream activations before and after critique, we introduce a measurable notion of "cognitive change." This connects prompting techniques directly with mechanistic interpretability, providing a novel framework for assessing the depth of LLM self-reflection.

### Experiment Justification
- **Experiment 1 (Activation Trajectory Tracking):** We need to extract residual stream activations across multiple rounds of self-critique to measure the magnitude and direction of representation drift (using cosine similarity).
- **Experiment 2 (Linear Separability of Correctness):** By training a linear probe to classify correct vs. incorrect reasoning on the activations, we can test whether self-critique makes the "correct" reasoning subspace more distinct and structured over time.

## Research Question
Does self-critique induce structured shifts in a model's internal representations (specifically in the residual stream), resulting in a more distinct and structured reasoning subspace, or does it merely cause shallow representational changes?

## Hypothesis Decomposition
1. **Representational Drift:** Self-critique causes a significant shift in residual stream activations compared to standard generation (measured via cosine distance).
2. **Subspace Structuring (Linear Separability):** The activations corresponding to correct reasoning steps become increasingly linearly separable from incorrect ones after self-critique.
3. **Convergence:** Meaningful reflection leads the activations to converge toward a stable "reasoning manifold" over multiple rounds, rather than diverging randomly.

## Proposed Methodology

### Approach
We will use a small, interpretable model (e.g., Qwen/Qwen2.5-1.5B or a small Llama model) and run it on reasoning tasks (GSM8K). We will extract internal activations from the residual streams during standard Chain-of-Thought (CoT) generation and during iterative self-critique prompting. We will then analyze these activations using cosine similarity (to measure drift) and linear probing (to measure separability).

### Experimental Steps
1. **Setup Environment & Models:** Initialize a HuggingFace model and tokenizer suitable for activation extraction.
2. **Data Processing:** Select a subset of GSM8K (e.g., 100-200 examples) to keep computational costs manageable.
3. **Activation Extraction (Baseline vs Critique):** 
   - Generate initial CoT answers. Extract residual activations at the final token of the reasoning.
   - Prompt the model to critique and revise its answer. Extract activations at the final token of the revision.
   - Repeat for 2-3 rounds of critique.
4. **Metric Computation (Drift):** Calculate the pairwise cosine similarity and L2 distance of activations across rounds for each layer.
5. **Metric Computation (Separability):** Train logistic regression probes on the activations at each layer to predict whether the final answer is correct. Compare probe accuracy/F1 across rounds.

### Baselines
- **Vanilla CoT (No Critique):** Activations from a standard single-pass generation.
- **Random Resampling:** Prompting the model to simply "generate another answer" without explicit critique, to distinguish true reflection from random variance.

### Evaluation Metrics
- **Representation Drift:** Cosine distance and L2 norm between activation vectors across reflection rounds.
- **Linear Separability:** Accuracy and F1 score of a linear probe trained to classify correct vs. incorrect answers based on layer activations.
- **Task Performance:** Final accuracy on GSM8K to correlate internal changes with external performance.

### Statistical Analysis Plan
- We will report the mean and standard deviation of cosine distances across the dataset.
- We will use paired t-tests to determine if the increase in probe accuracy after critique is statistically significant (p < 0.05).

## Expected Outcomes
We expect to see significant representation drift in mid-to-late layers during self-critique. If the hypothesis holds, the linear separability of correct answers should improve significantly in the post-critique activations compared to the initial CoT activations, indicating a more structured reasoning space.

## Timeline and Milestones
- **Phase 1 & 2 (Planning & Setup):** 30 mins
- **Phase 3 (Implementation of Extraction & Probing):** 60 mins
- **Phase 4 (Experiment Execution):** 60 mins
- **Phase 5 (Analysis):** 40 mins
- **Phase 6 (Documentation):** 30 mins

## Potential Challenges
- **Computational Cost:** Extracting activations for all layers on large datasets is memory-intensive. We will mitigate this by using a small model (e.g., 1.5B) and a subset of the data.
- **Ambiguous Critique:** The model might not actually change its answer during critique. We will filter for instances where the answer actually changes, or separately analyze "changed" vs "unchanged" trajectories.

## Success Criteria
The research is successful if we can definitively measure and statistically compare the representational drift and linear separability before and after self-critique, establishing whether internal restructuring occurs.