# Code Repositories for Representation Analysis

This directory contains external codebases for identifying and manipulating LLM internal states.

## Cloned Repositories

### 1. ProbingReflection
- **URL**: `https://github.com/xzAscC/ProbingReflection`
- **Purpose**: Implementation of "reflection vectors" from Zhu et al. (2025).
- **Location**: `code/probing_reflection/`
- **Key Files**:
  - `src/extract_reflection_vector.py`: Core logic for "difference-of-means" vector extraction.
  - `src/probe.py`: Probing for reflective states.
  - `models/`: Wrappers for HuggingFace models (Qwen, Llama).

### 2. ReflCtrl
- **URL**: `https://github.com/Trustworthy-ML-Lab/ReflCtrl`
- **Purpose**: Step-wise reflection steering using Representation Engineering (RepE).
- **Location**: `code/refl_ctrl/`
- **Key Files**:
  - `collect_activation.py`: Collect internal activations during reasoning.
  - `extract_dir.py`: Extract the "reflection direction" from activations.
  - `hook_utils.py`: Utilities for steering models via activations.

### 3. Factual Self-Awareness
- **URL**: `https://github.com/UKPLab/arxiv2025-self-awareness`
- **Purpose**: Linear probing for factual recall correctness.
- **Location**: `code/self_awareness/`
- **Key Files**:
  - `self_aware/probing/`: Training and evaluating probes for factual self-awareness.
  - `self_aware/data/`: Data loading for factual associations.

## Integration Plan

The experiment runner should:
1. Use `refl_ctrl` scripts to generate reasoning traces (CoT vs Reflection) and collect activations.
2. Use `probing_reflection` logic to compute the "reflection vector" at different critique stages.
3. Use `self_awareness` probes to measure if "internal conviction" shifts as the model critiques itself.
