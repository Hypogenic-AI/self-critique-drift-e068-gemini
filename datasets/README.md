# Datasets for Representation Drift Research

This directory contains datasets for evaluating LLM reasoning and internal self-awareness.

## Downloaded Datasets

### 1. GSM8K (Grade School Math 8K)
- **Source**: `gsm8k` (HuggingFace)
- **Task**: Multi-step mathematical reasoning.
- **Location**: `datasets/gsm8k/`
- **Why**: Standard benchmark to verify correctness and trigger self-reflection.

### 2. StrategyQA
- **Source**: `tasksource/strategy-qa` (HuggingFace)
- **Task**: Questions requiring implicit multi-step reasoning.
- **Location**: `datasets/strategy_qa/`
- **Why**: Tests model's ability to deliberate and revise reasoning chains.

### 3. ARC-Challenge (AI2 Reasoning Challenge)
- **Source**: `allenai/ai2_arc` (HuggingFace)
- **Task**: Grade-school level science questions.
- **Location**: `datasets/arc_challenge/`
- **Why**: Harder reasoning task than standard ARC.

### 4. Factual Recall Dataset
- **Source**: `UKPLab/arxiv2025-self-awareness` (GitHub)
- **Task**: Recall of (subject, relation, attribute) triplets.
- **Location**: `code/self_awareness/self_aware/data/`
- **Why**: Used to train "self-awareness" probes.

## Download Instructions

Datasets were downloaded using the `datasets` library:
```python
from datasets import load_dataset
# Example
ds = load_dataset("gsm8k", "main")
ds.save_to_disk("datasets/gsm8k")
```

## Sample Data (GSM8K)
```json
[
  {"question": "Natalia sold clips to 48 of her friends in April...", "answer": "Natalia sold 48/2 = 24 clips in May. Total = 48 + 24 = 72. #### 72"}
]
```
