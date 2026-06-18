"""
evaluate.py — Evaluation & Comparison: Baseline vs Fine-Tuned Model
====================================================================

This script runs a rigorous head-to-head comparison between the base
Mistral-7B-v0.3 model and the LoRA-fine-tuned version on medical Q&A.

WHY a dedicated evaluation script (not inline in train.py):
  - Training and evaluation have very different resource profiles: training
    uses gradient computation; evaluation is inference-only with torch.no_grad().
  - Separation lets you re-evaluate with different parameters (more samples,
    different temperature) without re-training.
  - Produces artifacts (markdown reports, JSON) that feed into plot_results.py
    and the final README.

HOW VRAM is managed:
  - Only ONE model is ever loaded at a time. After the baseline pass, we
    explicitly delete the model/tokenizer and call torch.cuda.empty_cache()
    before loading the fine-tuned version. This is critical for T4's 16GB limit.

METRICS (WHY each was chosen):
  - ROUGE-L: Measures longest common subsequence between generated and reference.
    Captures structural similarity — important for medical answers where key
    phrases must appear in the right order.
  - BLEU: Measures n-gram precision. The standard MT metric, useful here because
    medical Q&A has specific terminology that should appear verbatim.
  - Perplexity: exp(avg cross-entropy loss). The most fundamental LM metric —
    measures how "surprised" the model is by the reference answers. Lower =
    better. Unlike ROUGE/BLEU, this uses the model's own probability estimates,
    not just surface-level text matching.

USAGE:
  python evaluate.py                         # Uses defaults from config.yaml
  python evaluate.py --config my.yaml        # Custom config
  python evaluate.py --num-samples 50        # Override sample count
  python evaluate.py --skip-baseline         # Only evaluate fine-tuned model

Author: [Your Name]
Date: June 2026
"""

import os
import gc
import json
import math
import argparse
import logging
from typing import Dict, Any, List, Optional, Tuple

import yaml
import torch
import numpy as np
from tqdm import tqdm
from datasets import DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import PeftModel
import evaluate as hf_evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference Prompt Template
# ---------------------------------------------------------------------------
# WHY this must match the training template EXACTLY:
#   - Distribution shift: if the model was trained on "### Question:" but we
#     evaluate with "Question:", the model sees out-of-distribution input and
#     performance degrades. Consistency is critical.
#   - The template ends at "### Answer:\n" — the model generates everything
#     after this marker. We strip the prompt to extract only the answer.

INFERENCE_TEMPLATE = """Below is a medical question. Provide a thorough and accurate answer.

### Question:
{question}

### Answer:
"""


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    WHY YAML over hardcoded values:
      - Single source of truth for all hyperparameters across scripts.
      - Easy to version-control experiment configurations.
      - Avoids the "magic number" anti-pattern in production code.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary of configuration values.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the config file is malformed.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            f"Run from the project root or specify --config."
        )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def load_test_data(
    data_dir: str = "data/processed",
    num_samples: int = 100,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """
    Load and sample from the test split saved by data_prep.py.

    WHY sample instead of using the full test set:
      - Full test set (~1,700 rows) with generation takes 2-4 hours on T4.
      - 100 samples gives statistically meaningful ROUGE/BLEU in ~15-20 min.
      - Sampling with a fixed seed ensures reproducibility.

    WHY we extract question/answer pairs (not raw text):
      - The stored "text" column contains the full prompt+answer. We need to
        separate them to: (a) construct inference prompts, (b) compare
        generated answers against reference answers.

    Args:
        data_dir: Path to the processed dataset (Arrow format).
        num_samples: Number of test examples to evaluate on.
        seed: Random seed for reproducible sampling.

    Returns:
        List of dicts with 'question' and 'reference_answer' keys.

    Raises:
        FileNotFoundError: If the processed data directory doesn't exist.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Processed data not found at {data_dir}. "
            f"Run data_prep.py first."
        )

    logger.info(f"Loading test data from {data_dir}")
    splits = DatasetDict.load_from_disk(data_dir)

    if "test" not in splits:
        raise KeyError(
            f"No 'test' split found in {data_dir}. "
            f"Available splits: {list(splits.keys())}"
        )

    test_data = splits["test"]
    logger.info(f"Test set size: {len(test_data)} examples")

    # Cap num_samples at actual test set size
    actual_samples = min(num_samples, len(test_data))
    if actual_samples < num_samples:
        logger.warning(
            f"Requested {num_samples} samples but test set only has "
            f"{len(test_data)}. Using all {actual_samples}."
        )

    # Deterministic shuffle + select
    test_data = test_data.shuffle(seed=seed).select(range(actual_samples))
    logger.info(f"Sampled {actual_samples} examples for evaluation")

    # Parse question and answer from the formatted text
    # Format: "Below is a medical question...\n\n### Question:\n{q}\n\n### Answer:\n{a}"
    examples = []
    for row in test_data:
        text = row["text"]
        parsed = _parse_qa_from_text(text)
        if parsed is not None:
            examples.append(parsed)
        else:
            logger.warning(f"Could not parse Q&A from text: {text[:100]}...")

    logger.info(f"Successfully parsed {len(examples)} examples")
    return examples


def _parse_qa_from_text(text: str) -> Optional[Dict[str, str]]:
    """
    Extract question and answer from a formatted prompt string.

    WHY parse instead of using raw columns:
      - data_prep.py combines instruction+input into a single formatted text.
        The original columns may not perfectly reconstruct what the model sees.
      - By parsing the formatted text, we guarantee the question we use for
        inference is EXACTLY what the model was trained on (minus the answer).

    Args:
        text: Full formatted prompt string from the dataset.

    Returns:
        Dict with 'question' and 'reference_answer', or None if parsing fails.
    """
    question_marker = "### Question:\n"
    answer_marker = "### Answer:\n"

    q_start = text.find(question_marker)
    a_start = text.find(answer_marker)

    if q_start == -1 or a_start == -1:
        return None

    question = text[q_start + len(question_marker):a_start].strip()
    answer = text[a_start + len(answer_marker):].strip()

    if not question or not answer:
        return None

    return {"question": question, "reference_answer": answer}


def load_model_and_tokenizer(
    model_name: str,
    adapter_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model (optionally with LoRA adapter) in 4-bit quantization.

    WHY 4-bit quantization for evaluation (not just training):
      - Mistral 7B in fp16 = ~14GB. On T4 (16GB), there's no room for
        the KV cache needed during generation.
      - 4-bit reduces the model to ~4GB, leaving plenty of room for
        generation with long sequences.
      - QLoRA paper shows 4-bit inference quality matches fp16 within
        measurement noise for most tasks.

    WHY we load adapter separately (not merged):
      - Merging (base + adapter → single model) requires fp16 base weights,
        which won't fit in T4 VRAM alongside another model.
      - PeftModel keeps adapter weights separate, using almost no extra VRAM.
      - For deployment/export, merging is done in merge_and_export.py.

    Args:
        model_name: HuggingFace model identifier for the base model.
        adapter_path: Path to LoRA adapter directory. None = baseline model.
        config: Full config dict for quantization parameters.

    Returns:
        Tuple of (model, tokenizer).
    """
    config = config or {}

    # Build the quantization config — same as training for consistency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.get("load_in_4bit", True),
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=torch.float16,  # T4: NO bf16 support
    )

    model_label = "fine-tuned" if adapter_path else "baseline"
    logger.info(f"Loading {model_label} model: {model_name}")
    if adapter_path:
        logger.info(f"  LoRA adapter: {adapter_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Load LoRA adapter on top of the base model
    if adapter_path:
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"LoRA adapter not found at {adapter_path}. "
                f"Run train.py first to produce the adapter."
            )
        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info("LoRA adapter loaded successfully")

    # Put model in eval mode — disables dropout
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # WHY set pad_token = eos_token:
    #   - Mistral doesn't have a dedicated pad token. Without this,
    #     batch generation crashes with "pad_token_id is not set".
    #   - Using EOS as pad is standard practice for decoder-only models.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Model loaded. Device: {model.device}")
    return model, tokenizer


def unload_model(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
    """
    Explicitly free GPU memory after finishing with a model.

    WHY explicit cleanup (not just reassignment):
      - Python's garbage collector doesn't immediately free CUDA memory.
      - torch.cuda.empty_cache() returns memory to the CUDA allocator but
        doesn't help if Python still holds references.
      - We must: (1) delete Python references, (2) run GC, (3) empty cache.
      - On T4 with 16GB, failing to do this means the second model won't fit.

    Args:
        model: The model to unload.
        tokenizer: The tokenizer to unload.
    """
    logger.info("Unloading model and freeing VRAM...")
    del model
    del tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"VRAM after cleanup: {allocated:.2f} GB allocated")
    else:
        logger.info("No CUDA device — skipping cache cleanup")


def generate_answers(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: List[Dict[str, str]],
    config: Dict[str, Any],
    label: str = "model",
) -> List[str]:
    """
    Generate answers for all examples using the given model.

    WHY one-at-a-time generation (not batched):
      - With 4-bit quantization on T4, batch_size > 1 for generation risks OOM
        because the KV cache scales linearly with batch size.
      - One-at-a-time is slower (~1-2 examples/sec) but guarantees no OOM.
      - For 100 examples, this takes ~2-5 minutes — acceptable for evaluation.

    WHY torch.no_grad():
      - Evaluation doesn't need gradients. Disabling them saves ~50% VRAM
        (no gradient buffers) and speeds up inference by ~20%.

    WHY low temperature (0.1) for evaluation:
      - We want deterministic, factual answers for fair comparison.
      - High temperature adds randomness that makes metrics noisy.
      - 0.1 (not 0.0) avoids degenerate greedy decoding artifacts.

    Args:
        model: The loaded model (baseline or fine-tuned).
        tokenizer: The corresponding tokenizer.
        examples: List of dicts with 'question' keys.
        config: Config dict for generation parameters.
        label: Human-readable label for progress bar.

    Returns:
        List of generated answer strings (prompt stripped).
    """
    max_new_tokens = config.get("eval_max_new_tokens", 256)
    temperature = config.get("eval_temperature", 0.1)
    top_p = config.get("eval_top_p", 0.9)
    do_sample = config.get("eval_do_sample", True)

    # Build generation config
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )

    generated_answers = []

    with torch.no_grad():
        for example in tqdm(examples, desc=f"Generating ({label})", unit="ex"):
            prompt = INFERENCE_TEMPLATE.format(question=example["question"])

            try:
                answer = _generate_single(
                    model, tokenizer, prompt, gen_config
                )
            except Exception as e:
                logger.warning(
                    f"Generation failed for question: "
                    f"{example['question'][:80]}... Error: {e}"
                )
                answer = "[GENERATION FAILED]"

            generated_answers.append(answer)

    logger.info(
        f"Generated {len(generated_answers)} answers for {label}. "
        f"Failures: {sum(1 for a in generated_answers if a == '[GENERATION FAILED]')}"
    )
    return generated_answers


def _generate_single(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    gen_config: GenerationConfig,
) -> str:
    """
    Generate a single answer from a prompt.

    WHY we strip the prompt from the output:
      - model.generate() returns the FULL sequence (prompt + generated tokens).
      - We only want the answer, so we slice off the prompt tokens.
      - We also strip whitespace and handle edge cases (empty output, EOS only).

    Args:
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        prompt: The inference prompt (ends at "### Answer:\\n").
        gen_config: GenerationConfig with temperature, top_p, etc.

    Returns:
        The generated answer text (prompt stripped).
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    prompt_length = inputs["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs,
        generation_config=gen_config,
    )

    # Extract only the generated tokens (after the prompt)
    generated_ids = output_ids[0, prompt_length:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Handle edge cases
    if not answer:
        answer = "[EMPTY GENERATION]"

    # Truncate excessively long answers (safety net)
    max_answer_chars = 2000
    if len(answer) > max_answer_chars:
        answer = answer[:max_answer_chars] + "... [TRUNCATED]"

    return answer


def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: List[Dict[str, str]],
    config: Dict[str, Any],
    label: str = "model",
) -> float:
    """
    Compute perplexity on the test set using the model's own loss.

    WHY perplexity (in addition to ROUGE/BLEU):
      - ROUGE and BLEU measure surface-level text overlap. A model can score
        high on ROUGE by copying common phrases without understanding.
      - Perplexity measures the model's INTERNAL confidence — how likely it
        thinks the correct answer is. This is a deeper quality signal.
      - Perplexity = exp(average cross-entropy loss). Lower = model is less
        "surprised" by the correct answer = better understanding.

    WHY we compute loss on the FULL formatted text (prompt + answer):
      - We want to measure how well the model predicts the answer tokens
        given the question context. Using the full text with causal LM loss
        means the model's loss on the answer portion reflects its ability
        to generate correct medical answers.

    Args:
        model: Loaded model (in eval mode).
        tokenizer: Corresponding tokenizer.
        examples: List of dicts with 'question' and 'reference_answer' keys.
        config: Config dict for max_seq_length.
        label: Label for the progress bar.

    Returns:
        Perplexity (float). Lower is better.
    """
    max_seq_length = config.get("max_seq_length", 512)
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for example in tqdm(examples, desc=f"Perplexity ({label})", unit="ex"):
            # Reconstruct the full formatted text (prompt + answer)
            full_text = INFERENCE_TEMPLATE.format(
                question=example["question"]
            ) + example["reference_answer"]

            encodings = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
            )
            input_ids = encodings["input_ids"].to(model.device)

            # Skip very short sequences (edge case: tokenizer returns < 2 tokens)
            if input_ids.shape[1] < 2:
                continue

            # Forward pass with labels = input_ids (standard causal LM loss)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss.item()
            num_tokens = input_ids.shape[1] - 1  # loss is computed on n-1 tokens

            total_loss += loss * num_tokens
            total_tokens += num_tokens

    if total_tokens == 0:
        logger.error("No tokens processed for perplexity computation!")
        return float("inf")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    # Cap perplexity at a reasonable maximum to avoid overflow
    max_perplexity = 10000.0
    if perplexity > max_perplexity:
        logger.warning(
            f"Perplexity {perplexity:.2f} exceeds {max_perplexity}. "
            f"Capping at {max_perplexity}."
        )
        perplexity = max_perplexity

    logger.info(f"Perplexity ({label}): {perplexity:.2f} (avg loss: {avg_loss:.4f})")
    return perplexity


def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Compute ROUGE-L F1 score between predictions and references.

    WHY ROUGE-L (not ROUGE-1 or ROUGE-2):
      - ROUGE-1/2 measure unigram/bigram overlap — order doesn't matter.
      - ROUGE-L uses Longest Common Subsequence (LCS), which rewards answers
        that preserve the correct ORDER of medical terminology.
      - Example: "The patient has fever and cough" vs "cough and fever the patient has"
        would score similarly on ROUGE-1 but differently on ROUGE-L.
      - F1 variant balances precision (how much of the prediction is relevant)
        and recall (how much of the reference is covered).

    Args:
        predictions: List of generated answers.
        references: List of reference answers.

    Returns:
        Average ROUGE-L F1 score (0.0 to 1.0).
    """
    rouge = hf_evaluate.load("rouge")

    # Filter out failed generations for fair comparison
    valid_pairs = [
        (pred, ref)
        for pred, ref in zip(predictions, references)
        if pred not in ("[GENERATION FAILED]", "[EMPTY GENERATION]")
    ]

    if not valid_pairs:
        logger.warning("No valid predictions for ROUGE computation!")
        return 0.0

    valid_preds, valid_refs = zip(*valid_pairs)
    results = rouge.compute(
        predictions=list(valid_preds),
        references=list(valid_refs),
    )

    score = results["rougeL"]
    logger.info(f"ROUGE-L: {score:.4f} ({len(valid_pairs)} valid pairs)")
    return score


def compute_bleu(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Compute average sentence-level BLEU score.

    WHY BLEU for medical Q&A:
      - BLEU measures n-gram precision — how many n-grams in the generated
        answer also appear in the reference.
      - Medical answers contain specific terminology (drug names, anatomical
        terms, disease names) that should appear EXACTLY. BLEU rewards this.
      - We use sentence-level BLEU (not corpus-level) because each Q&A pair
        is independent, and we want per-example granularity.

    WHY smoothing (method1):
      - Standard BLEU gives 0.0 if ANY n-gram order has zero matches.
      - With short medical answers, 4-gram matches can be sparse.
      - SmoothingFunction().method1 adds epsilon to zero counts, avoiding
        the harsh zero penalty while still penalizing poor n-gram overlap.

    Args:
        predictions: List of generated answers.
        references: List of reference answers.

    Returns:
        Average sentence-level BLEU score (0.0 to 1.0).
    """
    smoother = SmoothingFunction().method1
    scores = []

    for pred, ref in zip(predictions, references):
        # Skip failed generations
        if pred in ("[GENERATION FAILED]", "[EMPTY GENERATION]"):
            continue

        # NLTK BLEU expects tokenized inputs (list of words)
        ref_tokens = ref.split()
        pred_tokens = pred.split()

        # Handle edge case: empty prediction or reference after splitting
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue

        try:
            score = sentence_bleu(
                [ref_tokens],  # list of reference tokenizations
                pred_tokens,
                smoothing_function=smoother,
            )
            scores.append(score)
        except Exception as e:
            logger.warning(f"BLEU computation failed: {e}")
            scores.append(0.0)

    if not scores:
        logger.warning("No valid predictions for BLEU computation!")
        return 0.0

    avg_bleu = np.mean(scores)
    logger.info(f"BLEU: {avg_bleu:.4f} ({len(scores)} valid pairs)")
    return float(avg_bleu)


def compute_improvement(baseline: float, finetuned: float, lower_is_better: bool = False) -> str:
    """
    Compute percentage improvement between baseline and fine-tuned scores.

    WHY signed percentage (not absolute difference):
      - Percentages are more intuitive: "+25.3%" instantly communicates impact.
      - The sign (+/-) shows direction: positive = fine-tuned is better.
      - For perplexity (lower is better), we flip the sign so positive still
        means improvement.

    Args:
        baseline: Baseline model's score.
        finetuned: Fine-tuned model's score.
        lower_is_better: If True, a decrease is an improvement (e.g., perplexity).

    Returns:
        Formatted string like "+25.3%" or "-12.1%".
    """
    if baseline == 0:
        return "N/A"

    if lower_is_better:
        # For perplexity: improvement = how much it decreased
        pct_change = ((baseline - finetuned) / baseline) * 100
    else:
        # For ROUGE/BLEU: improvement = how much it increased
        pct_change = ((finetuned - baseline) / baseline) * 100

    sign = "+" if pct_change >= 0 else ""
    return f"{sign}{pct_change:.1f}%"


def format_comparison_table(
    baseline_metrics: Dict[str, float],
    finetuned_metrics: Dict[str, float],
) -> str:
    """
    Format a markdown comparison table for stdout and file output.

    WHY markdown format:
      - Renders beautifully on GitHub, in Jupyter notebooks, and in the README.
      - Easy to copy-paste into project documentation.
      - Human-readable even as raw text in the terminal.

    Args:
        baseline_metrics: Dict with 'rouge_l', 'bleu', 'perplexity' keys.
        finetuned_metrics: Same structure for the fine-tuned model.

    Returns:
        Formatted markdown table string.
    """
    rouge_imp = compute_improvement(
        baseline_metrics["rouge_l"], finetuned_metrics["rouge_l"]
    )
    bleu_imp = compute_improvement(
        baseline_metrics["bleu"], finetuned_metrics["bleu"]
    )
    perp_imp = compute_improvement(
        baseline_metrics["perplexity"],
        finetuned_metrics["perplexity"],
        lower_is_better=True,
    )

    table = (
        "# Evaluation Results: Baseline vs Fine-Tuned\n\n"
        "| Metric     | Baseline | Fine-Tuned | Improvement |\n"
        "|------------|----------|------------|-------------|\n"
        f"| ROUGE-L    | {baseline_metrics['rouge_l']:.3f}    "
        f"| {finetuned_metrics['rouge_l']:.3f}      "
        f"| {rouge_imp:>11s} |\n"
        f"| BLEU       | {baseline_metrics['bleu']:.3f}    "
        f"| {finetuned_metrics['bleu']:.3f}      "
        f"| {bleu_imp:>11s} |\n"
        f"| Perplexity | {baseline_metrics['perplexity']:.2f}   "
        f"| {finetuned_metrics['perplexity']:.2f}     "
        f"| {perp_imp:>11s} |\n"
    )
    return table


def format_qualitative_examples(
    examples: List[Dict[str, str]],
    baseline_answers: List[str],
    finetuned_answers: List[str],
    num_examples: int = 10,
) -> str:
    """
    Format side-by-side comparison examples in markdown.

    WHY qualitative examples (not just numbers):
      - Metrics like ROUGE/BLEU are averages — they hide individual failures.
      - Seeing actual outputs reveals failure modes: does the baseline hallucinate?
        Does the fine-tuned model copy training data verbatim?
      - For portfolio projects, qualitative examples are more impressive than
        a ROUGE score to hiring managers who may not know what ROUGE-L = 0.42 means.

    Args:
        examples: List of dicts with 'question' and 'reference_answer'.
        baseline_answers: Generated answers from the baseline model.
        finetuned_answers: Generated answers from the fine-tuned model.
        num_examples: Number of examples to include.

    Returns:
        Formatted markdown string with side-by-side comparisons.
    """
    num_examples = min(num_examples, len(examples))

    lines = [
        "# Qualitative Comparison: Baseline vs Fine-Tuned\n",
        f"Showing {num_examples} randomly sampled examples.\n",
    ]

    for i in range(num_examples):
        ex = examples[i]
        baseline = baseline_answers[i] if i < len(baseline_answers) else "[N/A]"
        finetuned = finetuned_answers[i] if i < len(finetuned_answers) else "[N/A]"

        lines.extend([
            f"\n## Example {i + 1}\n",
            f"**Question:** {ex['question']}\n",
            f"**Reference Answer:** {ex['reference_answer']}\n",
            f"**Baseline Answer:** {baseline}\n",
            f"**Fine-Tuned Answer:** {finetuned}\n",
            "---\n",
        ])

    return "\n".join(lines)


def save_results(
    baseline_metrics: Dict[str, float],
    finetuned_metrics: Dict[str, float],
    baseline_answers: List[str],
    finetuned_answers: List[str],
    examples: List[Dict[str, str]],
    config: Dict[str, Any],
    output_dir: str = "results",
    examples_dir: str = "examples",
) -> None:
    """
    Save all evaluation results to disk.

    WHY three output formats:
      1. Markdown table (results/metrics_comparison.md):
         - For the README and GitHub rendering. Human-readable.
      2. JSON (results/eval_results.json):
         - Machine-readable. Fed into plot_results.py for visualizations.
         - Includes raw answers for post-hoc analysis.
      3. Qualitative examples (examples/comparison_examples.md):
         - Portfolio showcase. Demonstrates the model's actual behavior.

    Args:
        baseline_metrics: Baseline model metrics dict.
        finetuned_metrics: Fine-tuned model metrics dict.
        baseline_answers: Raw baseline answers for all examples.
        finetuned_answers: Raw fine-tuned answers for all examples.
        examples: The test examples with questions and references.
        config: Full config dict (saved for reproducibility).
        output_dir: Directory for metrics and JSON.
        examples_dir: Directory for qualitative examples.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(examples_dir, exist_ok=True)

    # 1. Markdown comparison table
    table = format_comparison_table(baseline_metrics, finetuned_metrics)
    table_path = os.path.join(output_dir, "metrics_comparison.md")
    with open(table_path, "w") as f:
        f.write(table)
    logger.info(f"Saved metrics table to {table_path}")

    # 2. Raw JSON results (for plot_results.py)
    json_results = {
        "config": {
            "model_name": config.get("model_name", ""),
            "num_eval_samples": config.get("num_eval_samples", 100),
            "eval_max_new_tokens": config.get("eval_max_new_tokens", 256),
            "eval_temperature": config.get("eval_temperature", 0.1),
            "eval_top_p": config.get("eval_top_p", 0.9),
        },
        "baseline": {
            "metrics": baseline_metrics,
            "answers": baseline_answers,
        },
        "finetuned": {
            "metrics": finetuned_metrics,
            "answers": finetuned_answers,
        },
        "examples": examples,
    }
    json_path = os.path.join(output_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved raw results to {json_path}")

    # 3. Qualitative examples
    num_qualitative = config.get("num_qualitative_examples", 10)
    qual_md = format_qualitative_examples(
        examples, baseline_answers, finetuned_answers, num_qualitative
    )
    qual_path = os.path.join(examples_dir, "comparison_examples.md")
    with open(qual_path, "w") as f:
        f.write(qual_md)
    logger.info(f"Saved {num_qualitative} qualitative examples to {qual_path}")


def find_adapter_path(output_dir: str = "results") -> str:
    """
    Auto-detect the LoRA adapter path within the training output directory.

    WHY auto-detect (not hardcode):
      - SFTTrainer/Trainer saves checkpoints with dynamic names like
        "checkpoint-7200" based on the number of training steps.
      - We look for the final checkpoint or the output_dir itself, since
        SFTTrainer saves the final adapter there when save_strategy="epoch".

    Args:
        output_dir: The training output directory (usually "results").

    Returns:
        Path to the adapter directory containing adapter_config.json.

    Raises:
        FileNotFoundError: If no adapter is found.
    """
    # Check 1: adapter directly in output_dir (SFTTrainer saves here by default)
    if os.path.exists(os.path.join(output_dir, "adapter_config.json")):
        logger.info(f"Found adapter at {output_dir}")
        return output_dir

    # Check 2: look for checkpoint-* subdirectories
    if os.path.exists(output_dir):
        checkpoints = sorted(
            [
                d
                for d in os.listdir(output_dir)
                if d.startswith("checkpoint-")
                and os.path.exists(
                    os.path.join(output_dir, d, "adapter_config.json")
                )
            ],
            key=lambda x: int(x.split("-")[1]),
        )
        if checkpoints:
            adapter_path = os.path.join(output_dir, checkpoints[-1])
            logger.info(f"Found adapter at {adapter_path} (latest checkpoint)")
            return adapter_path

    raise FileNotFoundError(
        f"No LoRA adapter found in {output_dir}. "
        f"Run train.py first, or specify --adapter-path explicitly."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """
    Main evaluation pipeline: load data → baseline pass → fine-tuned pass → compare.

    The pipeline loads only one model at a time and explicitly frees VRAM
    between passes. This is essential for T4 (16GB) where two quantized
    models cannot coexist in memory.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate and compare baseline vs fine-tuned medical Q&A model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override num_eval_samples from config",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Explicit path to LoRA adapter (default: auto-detect in results/)",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the baseline evaluation (only evaluate fine-tuned model)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save evaluation results (default: results/)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed data (default: data/processed/)",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Step 1: Load config and test data
    # -----------------------------------------------------------------------
    config = load_config(args.config)
    num_samples = args.num_samples or config.get("num_eval_samples", 100)
    config["num_eval_samples"] = num_samples  # Store effective value

    examples = load_test_data(
        data_dir=args.data_dir,
        num_samples=num_samples,
        seed=config.get("seed", 42),
    )
    references = [ex["reference_answer"] for ex in examples]

    model_name = config["model_name"]
    adapter_path = args.adapter_path or find_adapter_path(args.output_dir)

    # -----------------------------------------------------------------------
    # Step 2: BASELINE EVALUATION
    # -----------------------------------------------------------------------
    if not args.skip_baseline:
        logger.info("\n" + "=" * 60)
        logger.info("PASS 1/2: BASELINE MODEL (no fine-tuning)")
        logger.info("=" * 60)

        model, tokenizer = load_model_and_tokenizer(
            model_name, adapter_path=None, config=config
        )

        baseline_answers = generate_answers(
            model, tokenizer, examples, config, label="baseline"
        )
        baseline_perplexity = compute_perplexity(
            model, tokenizer, examples, config, label="baseline"
        )

        # Free VRAM before loading the next model
        unload_model(model, tokenizer)

        baseline_rouge = compute_rouge(baseline_answers, references)
        baseline_bleu = compute_bleu(baseline_answers, references)

        baseline_metrics = {
            "rouge_l": baseline_rouge,
            "bleu": baseline_bleu,
            "perplexity": baseline_perplexity,
        }
    else:
        logger.info("Skipping baseline evaluation (--skip-baseline)")
        baseline_answers = ["[SKIPPED]"] * len(examples)
        baseline_metrics = {
            "rouge_l": 0.0,
            "bleu": 0.0,
            "perplexity": float("inf"),
        }

    # -----------------------------------------------------------------------
    # Step 3: FINE-TUNED MODEL EVALUATION
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("PASS 2/2: FINE-TUNED MODEL (with LoRA adapter)")
    logger.info("=" * 60)

    model, tokenizer = load_model_and_tokenizer(
        model_name, adapter_path=adapter_path, config=config
    )

    finetuned_answers = generate_answers(
        model, tokenizer, examples, config, label="fine-tuned"
    )
    finetuned_perplexity = compute_perplexity(
        model, tokenizer, examples, config, label="fine-tuned"
    )

    # Free VRAM — done with inference
    unload_model(model, tokenizer)

    finetuned_rouge = compute_rouge(finetuned_answers, references)
    finetuned_bleu = compute_bleu(finetuned_answers, references)

    finetuned_metrics = {
        "rouge_l": finetuned_rouge,
        "bleu": finetuned_bleu,
        "perplexity": finetuned_perplexity,
    }

    # -----------------------------------------------------------------------
    # Step 4: Display comparison and save results
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)

    table = format_comparison_table(baseline_metrics, finetuned_metrics)
    print("\n" + table)

    save_results(
        baseline_metrics=baseline_metrics,
        finetuned_metrics=finetuned_metrics,
        baseline_answers=baseline_answers,
        finetuned_answers=finetuned_answers,
        examples=examples,
        config=config,
        output_dir=args.output_dir,
        examples_dir="examples",
    )

    logger.info("\n✅ Evaluation complete!")
    logger.info(f"   Metrics table:  {args.output_dir}/metrics_comparison.md")
    logger.info(f"   Raw JSON:       {args.output_dir}/eval_results.json")
    logger.info(f"   Examples:       examples/comparison_examples.md")
    logger.info(f"   Next step:      python plot_results.py")


if __name__ == "__main__":
    main()
