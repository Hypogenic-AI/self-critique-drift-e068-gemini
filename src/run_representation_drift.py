#!/usr/bin/env python3
"""Run representation-drift experiments under self-critique.

This script executes:
1) Local mechanistic experiments on HF causal LMs (hidden states + behavior)
2) Optional OpenAI API behavioral triangulation (no hidden states)
3) Statistical analysis and plot generation
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Avoid torch inductor cache path resolution failures in containerized uid contexts.
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str((Path.cwd() / ".cache" / "torchinductor").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import load_from_disk
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, silhouette_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Sample:
    dataset: str
    question: str
    reference: str
    choices: Dict[str, str] | None = None


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_samples(data_root: Path, n_gsm: int, n_csqa: int, seed: int) -> List[Sample]:
    rng = random.Random(seed)

    gsm = load_from_disk(str(data_root / "gsm8k"))["test"]
    gsm_idx = list(range(len(gsm)))
    rng.shuffle(gsm_idx)
    gsm_samples: List[Sample] = []
    for i in gsm_idx[:n_gsm]:
        row = gsm[i]
        m = re.search(r"####\s*([-+]?\d[\d,\.]*)", row["answer"])
        ref = m.group(1).replace(",", "") if m else ""
        gsm_samples.append(Sample(dataset="gsm8k", question=row["question"], reference=ref))

    csqa = load_from_disk(str(data_root / "commonsense_qa"))["validation"]
    cs_idx = list(range(len(csqa)))
    rng.shuffle(cs_idx)
    cs_samples: List[Sample] = []
    for i in cs_idx[:n_csqa]:
        row = csqa[i]
        choices = {k: v for k, v in zip(row["choices"]["label"], row["choices"]["text"])}
        cs_samples.append(
            Sample(
                dataset="commonsense_qa",
                question=row["question"],
                reference=row.get("answerKey", ""),
                choices=choices,
            )
        )

    return gsm_samples + cs_samples


def format_question(sample: Sample) -> str:
    if sample.dataset == "commonsense_qa":
        choice_txt = "\n".join([f"{k}. {v}" for k, v in sample.choices.items()])
        return f"Question: {sample.question}\nChoices:\n{choice_txt}"
    return f"Question: {sample.question}"


def build_prompt(task: str, style: str, question_text: str, draft: str = "", critique: str = "") -> str:
    answer_fmt = (
        "End with exactly one line: FINAL: <number>."
        if task == "gsm8k"
        else "End with exactly one line: FINAL: <A/B/C/D/E>."
    )

    if style == "direct":
        return (
            "You are a careful reasoner. Solve the problem step by step briefly, then provide the final answer.\n"
            f"{answer_fmt}\n\n{question_text}"
        )
    if style == "rewrite":
        return (
            "Rewrite the draft answer for clarity and concision only. Preserve the reasoning intent.\n"
            f"{answer_fmt}\n\n{question_text}\n\nDraft answer:\n{draft}"
        )
    if style == "self_critique":
        return (
            "Critique the draft answer. Identify logical or arithmetic errors and missing steps."
            " Do not provide a final answer yet.\n\n"
            f"{question_text}\n\nDraft answer:\n{draft}"
        )
    if style == "revise":
        return (
            "Revise the draft answer using the critique. Provide corrected reasoning and final answer.\n"
            f"{answer_fmt}\n\n{question_text}\n\nDraft answer:\n{draft}\n\nCritique:\n{critique}"
        )
    raise ValueError(f"Unknown style: {style}")


def external_critique_template(task: str) -> str:
    if task == "gsm8k":
        return (
            "Check arithmetic carefully, ensure each intermediate equation is valid, and verify units."
            " The final line must contain only the computed numeric answer."
        )
    return (
        "Re-evaluate each option against the question constraints and remove options with weak semantic fit."
        " Choose the single most plausible option letter."
    )


def parse_prediction(sample: Sample, text: str) -> str:
    line_match = re.findall(r"FINAL\s*:\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    if sample.dataset == "gsm8k":
        if line_match:
            m = re.search(r"[-+]?\d[\d,\.]*", line_match[-1])
            if m:
                return m.group(0).replace(",", "")
        nums = re.findall(r"[-+]?\d[\d,\.]*", text)
        return nums[-1].replace(",", "") if nums else ""
    if line_match:
        m = re.search(r"\b([A-E])\b", line_match[-1].upper())
        if m:
            return m.group(1)
    m2 = re.findall(r"\b([A-E])\b", text.upper())
    return m2[-1] if m2 else ""


def is_correct(sample: Sample, pred: str) -> int:
    if not pred:
        return 0
    if sample.dataset == "gsm8k":
        try:
            return int(float(pred) == float(sample.reference))
        except Exception:
            return 0
    return int(pred.strip().upper() == sample.reference.strip().upper())


def maybe_chat_format(tokenizer: AutoTokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        msgs = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return prompt
    return prompt


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    prompt_text = maybe_chat_format(tokenizer, prompt)
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=0.95 if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def get_layer_vectors(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    selected_layers: List[int],
) -> Dict[int, np.ndarray]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    hidden_states = outputs.hidden_states
    vectors: Dict[int, np.ndarray] = {}
    for li in selected_layers:
        h = hidden_states[li][0, -1, :].detach().float().cpu().numpy()
        vectors[li] = h
    return vectors


def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return 0.0
    return float(1.0 - np.dot(x, y) / (nx * ny))


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    # x, y: [n, d]
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    k = x @ x.T
    l = y @ y.T
    hsic = np.sum(k * l)
    n1 = np.linalg.norm(k, ord="fro")
    n2 = np.linalg.norm(l, ord="fro")
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(hsic / (n1 * n2))


def paired_test(a: List[float], b: List[float]) -> Dict[str, float | str]:
    a_np = np.array(a)
    b_np = np.array(b)
    diff = a_np - b_np
    if len(diff) < 3:
        return {"test": "insufficient", "p": 1.0, "effect": 0.0}
    sw_p = stats.shapiro(diff).pvalue if len(diff) <= 5000 else 0.0
    if sw_p > 0.05:
        t = stats.ttest_rel(a_np, b_np)
        d = float(diff.mean() / (diff.std(ddof=1) + 1e-8))
        return {"test": "paired_t", "p": float(t.pvalue), "effect": d}
    w = stats.wilcoxon(a_np, b_np, zero_method="wilcox", correction=True)
    d = float(np.median(diff))
    return {"test": "wilcoxon", "p": float(w.pvalue), "effect": d}


def bootstrap_ci(values: List[float], n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    arr = np.array(values)
    if len(arr) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(sample.mean())
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def benjamini_hochberg(pvals: List[float]) -> List[float]:
    n = len(pvals)
    idx = np.argsort(pvals)
    q = np.zeros(n)
    prev = 1.0
    for rank, i in enumerate(idx[::-1], start=1):
        k = n - rank + 1
        val = min(prev, pvals[i] * n / k)
        q[i] = val
        prev = val
    return q.tolist()


def run_local_model_experiment(
    model_id: str,
    samples: List[Sample],
    results_dir: Path,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> Dict:
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    selected = sorted(set([1, max(1, n_layers // 4), max(1, n_layers // 2), max(1, (3 * n_layers) // 4), n_layers]))

    # Storage
    metrics = {
        "accuracy": defaultdict(list),
        "drift": defaultdict(lambda: defaultdict(list)),
        "step_drift": defaultdict(lambda: defaultdict(list)),
    }
    probe_vectors = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # probe_vectors[condition][stage][layer] -> [vectors]
    probe_labels = defaultdict(lambda: defaultdict(list))

    raw_records = []

    for idx, sample in enumerate(samples):
        qtxt = format_question(sample)

        # Shared first draft for all conditions
        p0 = build_prompt(sample.dataset, "direct", qtxt)
        draft = generate_text(model, tokenizer, p0, max_new_tokens=max_new_tokens, temperature=temperature)
        draft_pred = parse_prediction(sample, draft)
        draft_ok = is_correct(sample, draft_pred)
        draft_vec = get_layer_vectors(model, tokenizer, draft, selected)

        # Direct condition
        metrics["accuracy"]["direct"].append(draft_ok)
        for li in selected:
            probe_vectors["direct"]["pre"][li].append(draft_vec[li])
            probe_labels["direct"]["pre"].append(draft_ok)

        # Rewrite control
        rewrite = generate_text(
            model,
            tokenizer,
            build_prompt(sample.dataset, "rewrite", qtxt, draft=draft),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        rw_pred = parse_prediction(sample, rewrite)
        rw_ok = is_correct(sample, rw_pred)
        rw_vec = get_layer_vectors(model, tokenizer, rewrite, selected)
        metrics["accuracy"]["rewrite_control"].append(rw_ok)
        for li in selected:
            d = cosine_distance(draft_vec[li], rw_vec[li])
            metrics["drift"]["rewrite_control"][li].append(d)
            probe_vectors["rewrite_control"]["pre"][li].append(draft_vec[li])
            probe_vectors["rewrite_control"]["post"][li].append(rw_vec[li])
        probe_labels["rewrite_control"]["pre"].append(draft_ok)
        probe_labels["rewrite_control"]["post"].append(rw_ok)

        # Self-critique single round
        critique1 = generate_text(
            model,
            tokenizer,
            build_prompt(sample.dataset, "self_critique", qtxt, draft=draft),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        revise1 = generate_text(
            model,
            tokenizer,
            build_prompt(sample.dataset, "revise", qtxt, draft=draft, critique=critique1),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        s1_pred = parse_prediction(sample, revise1)
        s1_ok = is_correct(sample, s1_pred)
        s1_vec = get_layer_vectors(model, tokenizer, revise1, selected)
        metrics["accuracy"]["self_critique_1"].append(s1_ok)
        for li in selected:
            d = cosine_distance(draft_vec[li], s1_vec[li])
            metrics["drift"]["self_critique_1"][li].append(d)
            probe_vectors["self_critique_1"]["pre"][li].append(draft_vec[li])
            probe_vectors["self_critique_1"]["post"][li].append(s1_vec[li])
        probe_labels["self_critique_1"]["pre"].append(draft_ok)
        probe_labels["self_critique_1"]["post"].append(s1_ok)

        # Self-critique multi-round (3 rounds total revisions)
        round_vecs = [draft_vec]
        current_answer = revise1
        current_vec = s1_vec
        round_answers = [draft, revise1]
        round_correct = [draft_ok, s1_ok]

        for _ in range(2):
            crit = generate_text(
                model,
                tokenizer,
                build_prompt(sample.dataset, "self_critique", qtxt, draft=current_answer),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            current_answer = generate_text(
                model,
                tokenizer,
                build_prompt(sample.dataset, "revise", qtxt, draft=current_answer, critique=crit),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            pred = parse_prediction(sample, current_answer)
            ok = is_correct(sample, pred)
            current_vec = get_layer_vectors(model, tokenizer, current_answer, selected)
            round_answers.append(current_answer)
            round_correct.append(ok)
            round_vecs.append(current_vec)

        # Add first revised vec to trajectory at index 1
        round_vecs.insert(1, s1_vec)
        final_m = round_correct[-1]
        metrics["accuracy"]["self_critique_3"].append(final_m)

        for li in selected:
            d0 = cosine_distance(round_vecs[0][li], round_vecs[-1][li])
            metrics["drift"]["self_critique_3"][li].append(d0)
            # Step drifts for convergence analysis
            for r in range(1, len(round_vecs)):
                sd = cosine_distance(round_vecs[r - 1][li], round_vecs[r][li])
                metrics["step_drift"][f"round{r}"][li].append(sd)

            probe_vectors["self_critique_3"]["pre"][li].append(round_vecs[0][li])
            probe_vectors["self_critique_3"]["post"][li].append(round_vecs[-1][li])
        probe_labels["self_critique_3"]["pre"].append(round_correct[0])
        probe_labels["self_critique_3"]["post"].append(round_correct[-1])

        # External critique condition
        ext = external_critique_template(sample.dataset)
        ext_rev = generate_text(
            model,
            tokenizer,
            build_prompt(sample.dataset, "revise", qtxt, draft=draft, critique=ext),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        ex_pred = parse_prediction(sample, ext_rev)
        ex_ok = is_correct(sample, ex_pred)
        ex_vec = get_layer_vectors(model, tokenizer, ext_rev, selected)
        metrics["accuracy"]["external_critique"].append(ex_ok)
        for li in selected:
            d = cosine_distance(draft_vec[li], ex_vec[li])
            metrics["drift"]["external_critique"][li].append(d)
            probe_vectors["external_critique"]["pre"][li].append(draft_vec[li])
            probe_vectors["external_critique"]["post"][li].append(ex_vec[li])
        probe_labels["external_critique"]["pre"].append(draft_ok)
        probe_labels["external_critique"]["post"].append(ex_ok)

        raw_records.append(
            {
                "idx": idx,
                "dataset": sample.dataset,
                "reference": sample.reference,
                "direct_pred": draft_pred,
                "direct_correct": draft_ok,
                "rewrite_correct": rw_ok,
                "self1_correct": s1_ok,
                "self3_correct": final_m,
                "external_correct": ex_ok,
            }
        )

        if (idx + 1) % 10 == 0:
            print(f"[{model_id}] processed {idx+1}/{len(samples)}")

    # Aggregate summary stats
    summary = {
        "model_id": model_id,
        "n_samples": len(samples),
        "layers_selected": selected,
        "accuracy": {},
        "drift": {},
        "step_drift": {},
        "probe": {},
        "manifold": {},
        "stats": {},
        "runtime_seconds": time.time() - start,
    }

    for cond, vals in metrics["accuracy"].items():
        arr = np.array(vals)
        summary["accuracy"][cond] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "n": int(len(arr)),
            "ci95": bootstrap_ci(vals),
        }

    # Paired stats vs direct accuracy
    direct_vals = metrics["accuracy"]["direct"]
    pvals = []
    stats_map = {}
    for cond in ["rewrite_control", "self_critique_1", "self_critique_3", "external_critique"]:
        st = paired_test(metrics["accuracy"][cond], direct_vals)
        stats_map[f"accuracy_{cond}_vs_direct"] = st
        pvals.append(st["p"])
    qvals = benjamini_hochberg(pvals)
    for qi, cond in enumerate(["rewrite_control", "self_critique_1", "self_critique_3", "external_critique"]):
        stats_map[f"accuracy_{cond}_vs_direct"]["q_fdr"] = qvals[qi]
    summary["stats"].update(stats_map)

    # Drift summary and CKA
    for cond in ["rewrite_control", "self_critique_1", "self_critique_3", "external_critique"]:
        summary["drift"][cond] = {}
        for li in selected:
            vals = metrics["drift"][cond][li]
            summary["drift"][cond][str(li)] = {
                "mean": float(np.mean(vals)) if vals else 0.0,
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "ci95": bootstrap_ci(vals),
            }

    for round_name, by_layer in metrics["step_drift"].items():
        summary["step_drift"][round_name] = {}
        for li in selected:
            vals = by_layer[li]
            summary["step_drift"][round_name][str(li)] = float(np.mean(vals)) if vals else 0.0

    # Probe + manifold metrics
    for cond in ["rewrite_control", "self_critique_1", "self_critique_3", "external_critique"]:
        summary["probe"][cond] = {}
        summary["manifold"][cond] = {}
        for stage in ["pre", "post"]:
            summary["probe"][cond][stage] = {}
            summary["manifold"][cond][stage] = {}
            y = np.array(probe_labels[cond][stage])
            for li in selected:
                x = np.array(probe_vectors[cond][stage][li])
                if len(np.unique(y)) < 2 or len(y) < 10:
                    auc = float("nan")
                    f1 = float("nan")
                    sil = float("nan")
                else:
                    unique, counts = np.unique(y, return_counts=True)
                    min_count = counts.min() if len(counts) > 0 else 0
                    stratify_y = y if min_count >= 2 else None
                    x_train, x_test, y_train, y_test = train_test_split(
                        x, y, test_size=0.3, random_state=seed, stratify=stratify_y
                    )
                    clf = LogisticRegression(max_iter=400)
                    clf.fit(x_train, y_train)
                    probs = clf.predict_proba(x_test)[:, 1]
                    preds = (probs >= 0.5).astype(int)
                    auc = float(roc_auc_score(y_test, probs))
                    f1 = float(f1_score(y_test, preds))

                    # Manifold geometry in 2D PCA space
                    pca = PCA(n_components=2, random_state=seed)
                    z = pca.fit_transform(x)
                    sil = float(silhouette_score(z, y)) if len(np.unique(y)) > 1 else float("nan")

                summary["probe"][cond][stage][str(li)] = {"auc": auc, "f1": f1}
                summary["manifold"][cond][stage][str(li)] = {"silhouette": sil}

        # CKA pre vs post by layer
        summary.setdefault("cka", {}).setdefault(cond, {})
        for li in selected:
            x = np.array(probe_vectors[cond]["pre"][li])
            y = np.array(probe_vectors[cond]["post"][li])
            summary["cka"][cond][str(li)] = linear_cka(x, y) if len(x) == len(y) and len(x) > 1 else float("nan")

    # Save outputs
    out_json = results_dir / f"metrics_{sanitize_model_id(model_id)}.json"
    out_raw = results_dir / f"raw_{sanitize_model_id(model_id)}.jsonl"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_raw, "w", encoding="utf-8") as f:
        for r in raw_records:
            f.write(json.dumps(r) + "\n")

    make_plots(summary, results_dir / "plots", model_id)
    return summary


def sanitize_model_id(mid: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", mid)


def make_plots(summary: Dict, plots_dir: Path, model_id: str) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Accuracy bar chart
    acc_items = summary["accuracy"]
    labels = list(acc_items.keys())
    means = [acc_items[k]["mean"] for k in labels]
    plt.figure(figsize=(8, 4))
    sns.barplot(x=labels, y=means)
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy by Condition ({model_id})")
    plt.tight_layout()
    plt.savefig(plots_dir / f"accuracy_{sanitize_model_id(model_id)}.png", dpi=160)
    plt.close()

    # Drift by layer for key conditions
    drift = summary.get("drift", {})
    plt.figure(figsize=(8, 4))
    for cond in ["rewrite_control", "self_critique_1", "self_critique_3", "external_critique"]:
        if cond not in drift:
            continue
        layers = sorted([int(k) for k in drift[cond].keys()])
        vals = [drift[cond][str(k)]["mean"] for k in layers]
        plt.plot(layers, vals, marker="o", label=cond)
    plt.xlabel("Layer index")
    plt.ylabel("Mean cosine drift from draft")
    plt.title(f"Layer-wise Representation Drift ({model_id})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / f"drift_layers_{sanitize_model_id(model_id)}.png", dpi=160)
    plt.close()

    # Multi-round step drift convergence
    step = summary.get("step_drift", {})
    if step:
        round_ids = sorted(step.keys(), key=lambda x: int(x.replace("round", "")))
        last_layer = str(max(summary["layers_selected"]))
        vals = [step[r].get(last_layer, np.nan) for r in round_ids]
        plt.figure(figsize=(6, 4))
        plt.plot(round_ids, vals, marker="o")
        plt.ylabel(f"Step drift (layer {last_layer})")
        plt.title(f"Trajectory Stability Across Reflection Rounds ({model_id})")
        plt.tight_layout()
        plt.savefig(plots_dir / f"trajectory_{sanitize_model_id(model_id)}.png", dpi=160)
        plt.close()


def run_openai_behavioral(samples: List[Sample], results_dir: Path, max_items: int = 20) -> Dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"status": "skipped", "reason": "OPENAI_API_KEY missing"}

    try:
        from openai import OpenAI
    except Exception as e:
        return {"status": "skipped", "reason": f"openai package missing: {e}"}

    client = OpenAI(api_key=api_key)
    model_name = "gpt-4.1"

    picked = samples[:max_items]
    direct_acc, self_acc = [], []
    calls = 0

    for s in picked:
        qtxt = format_question(s)

        # Direct
        p_direct = build_prompt(s.dataset, "direct", qtxt)
        r0 = client.responses.create(model=model_name, input=p_direct, temperature=0)
        t0 = r0.output_text
        pred0 = parse_prediction(s, t0)
        ok0 = is_correct(s, pred0)

        # Self-critique single
        p_crit = build_prompt(s.dataset, "self_critique", qtxt, draft=t0)
        rc = client.responses.create(model=model_name, input=p_crit, temperature=0)
        crit = rc.output_text
        p_rev = build_prompt(s.dataset, "revise", qtxt, draft=t0, critique=crit)
        rr = client.responses.create(model=model_name, input=p_rev, temperature=0)
        t1 = rr.output_text
        pred1 = parse_prediction(s, t1)
        ok1 = is_correct(s, pred1)

        direct_acc.append(ok0)
        self_acc.append(ok1)
        calls += 3

    st = paired_test(self_acc, direct_acc)
    return {
        "status": "ok",
        "model": model_name,
        "n_samples": len(picked),
        "api_calls": calls,
        "direct_accuracy": float(np.mean(direct_acc)) if direct_acc else 0.0,
        "self_critique_accuracy": float(np.mean(self_acc)) if self_acc else 0.0,
        "comparison": st,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="datasets")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-gsm", type=int, default=30)
    parser.add_argument("--n-csqa", type=int, default=30)
    parser.add_argument("--n-gsm-small", type=int, default=20)
    parser.add_argument("--n-csqa-small", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--main-model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument(
        "--small-model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument("--skip-openai", action="store_true")
    parser.add_argument("--skip-main", action="store_true")
    parser.add_argument("--skip-small", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    data_root = Path(args.data_root)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "plots").mkdir(parents=True, exist_ok=True)

    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_count": torch.cuda.device_count(),
        "devices": [],
    }
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gpu_info["devices"].append({"name": p.name, "memory_gb": round(p.total_memory / (1024**3), 2)})

    samples_main = load_samples(data_root, args.n_gsm, args.n_csqa, args.seed)
    samples_small = load_samples(data_root, args.n_gsm_small, args.n_csqa_small, args.seed + 1)

    env = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "seed": args.seed,
        "python": os.sys.version,
        "torch": torch.__version__,
        "gpu": gpu_info,
        "config": vars(args),
    }
    with open(results_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)

    main_summary = {"status": "skipped"}
    small_summary = {"status": "skipped"}

    if not args.skip_main:
        print("Running main model experiment...")
        main_summary = run_local_model_experiment(
            args.main_model,
            samples_main,
            results_dir,
            args.max_new_tokens,
            args.temperature,
            args.seed,
        )

    if not args.skip_small:
        print("Running small model experiment...")
        small_summary = run_local_model_experiment(
            args.small_model,
            samples_small,
            results_dir,
            args.max_new_tokens,
            args.temperature,
            args.seed,
        )

    openai_summary = {"status": "skipped", "reason": "flag"}
    if not args.skip_openai:
        openai_summary = run_openai_behavioral(samples_main, results_dir, max_items=min(20, len(samples_main)))

    combined = {
        "environment": env,
        "main_model": main_summary,
        "small_model": small_summary,
        "openai_behavioral": openai_summary,
    }
    with open(results_dir / "metrics_combined.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print("Done. Wrote results to", results_dir)


if __name__ == "__main__":
    main()
