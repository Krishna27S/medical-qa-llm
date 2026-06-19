"""
train.py — QLoRA Fine-Tuning Script for Medical Q&A
====================================================

Fine-tunes Mistral-7B-v0.3 on a medical Q&A dataset using QLoRA (4-bit
quantization + Low-Rank Adaptation).  The entire training loop is configured
through config.yaml — no hyperparameters are hardcoded in this file.

KEY DESIGN DECISIONS:
   1. QLoRA over full fine-tuning: Mistral 7B in fp16 ≈ 14 GB VRAM.  With
      optimizer states and activations a T4 (16 GB) cannot fit it.  QLoRA
      compresses the base model to ~4 GB, and only ~20 M LoRA parameters
      are trained in fp32 (small enough to fit easily).
   2. NO mixed precision (fp16=False, bf16=False): Mistral 7B stores weights
      in bf16 on HuggingFace. The PyTorch gradient scaler (used with fp16)
      crashes on bf16 tensors. Since QLoRA already compresses the model to
      ~4 GB, mixed precision training is redundant for memory savings.
  3. Gradient checkpointing: Trades ~20 % wall-clock time for ~40 % VRAM
     savings.  Without it the backward pass OOMs on T4.
  4. SFTTrainer from trl: Purpose-built for supervised fine-tuning of causal
     LMs.  Handles tokenisation, packing, and causal-LM loss automatically,
     removing boilerplate and reducing bugs.

USAGE:
  python train.py                        # Full training run
  python train.py --config my.yaml       # Custom config
  python train.py --dry-run              # 3 steps only (for testing)

Author: [Your Name]
Date: June 2026
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, Any, Tuple

import yaml
import torch
from datasets import DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer

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
# Configuration
# ---------------------------------------------------------------------------
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load the central YAML config file.

    WHY a single config file:
      - Every hyperparameter is tracked in one place, making experiments
        reproducible and version-controllable via git.
      - Python code never needs editing to tweak a run — only config.yaml.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary of all configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Run from the project root or pass --config <path>."
        )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------
def create_bnb_config(config: Dict[str, Any]) -> BitsAndBytesConfig:
    """
    Build the BitsAndBytesConfig for 4-bit QLoRA quantization.

    WHY these specific settings:
      - load_in_4bit: Compresses 7 B parameters from fp16 (14 GB) to ~4 GB,
        fitting comfortably on a 16 GB T4 alongside optimizer states and
        activation memory.
      - nf4: NormalFloat4 is information-theoretically optimal for weights
        that follow a roughly Gaussian distribution (Dettmers et al., 2023).
      - double_quant: Quantises the quantisation constants themselves, saving
        an additional ~0.4 GB with no quality loss.  Free VRAM.
      - float16 compute dtype: T4 has hardware fp16 tensor cores (compute
        capability 7.5) but NO native bf16 support.  Using bf16 here would
        fall back to slow software emulation and risk numerical instability.

    Args:
        config: Project configuration dictionary.

    Returns:
        Configured BitsAndBytesConfig instance.
    """
    # Map the string dtype from config to a torch dtype.
    # IMPORTANT: Must be torch.float16 on T4.  torch.bfloat16 will silently
    # degrade performance or cause NaN losses.
    compute_dtype_str = config.get("bnb_4bit_compute_dtype", "float16")
    if compute_dtype_str == "float16":
        compute_dtype = torch.float16
    elif compute_dtype_str == "bfloat16":
        logger.warning(
            "⚠️  bfloat16 selected but T4 has NO native bf16 support. "
            "Overriding to float16 for safety."
        )
        compute_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported compute dtype: {compute_dtype_str}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.get("load_in_4bit", True),
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype,
    )

    logger.info(
        f"BitsAndBytes config: 4-bit={bnb_config.load_in_4bit}, "
        f"quant_type={bnb_config.bnb_4bit_quant_type}, "
        f"double_quant={bnb_config.bnb_4bit_use_double_quant}, "
        f"compute_dtype={compute_dtype}"
    )
    return bnb_config


# ---------------------------------------------------------------------------
# Model & Tokenizer
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(
    config: Dict[str, Any],
    bnb_config: BitsAndBytesConfig,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the base model (quantized) and its tokenizer.

    WHY device_map="auto":
      - Automatically places model layers across available devices (GPU, CPU,
        disk) using the accelerate library's infer_auto_device_map.
      - On a single-GPU Colab notebook this simply puts everything on the T4.
      - If VRAM is tight, layers spill to CPU RAM gracefully.

    WHY pad_token = eos_token:
      - Mistral's tokenizer has no dedicated pad token.  Without one,
        batched training crashes.  Using eos_token as the pad token is the
        standard workaround for decoder-only models.
      - Since we use labels=-100 for padding positions, the model never
        learns to predict the pad token — so reusing eos is harmless.

    WHY padding_side = "right":
      - For causal (left-to-right) generation, padding should go on the
        right so that the actual tokens are contiguous from position 0.
      - LoRA training specifically requires right-padding to avoid
        misaligned position encodings.

    Args:
        config: Project configuration dictionary.
        bnb_config: Quantization configuration.

    Returns:
        Tuple of (model, tokenizer).
    """
    model_name = config["model_name"]
    logger.info(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,  # Force float16 for non-quantized layers
    )

    # CRITICAL: Force-cast ALL remaining bfloat16 tensors to float16.
    # WHY: Mistral 7B on HuggingFace stores weights in bf16. Under 4-bit
    # quantization, most weights become int4, but non-quantized parts
    # (layer norms, embeddings, lm_head) keep bf16. Even with fp16=False
    # in training, some internal operations may choke on mixed bf16/fp16.
    # Belt-and-suspenders: cast everything that's bf16 to fp16.
    bf16_count = 0
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)
            bf16_count += 1
    for name, buf in model.named_buffers():
        if buf.dtype == torch.bfloat16:
            buf.data = buf.data.to(torch.float16)
            bf16_count += 1
    if bf16_count > 0:
        logger.info(f"Cast {bf16_count} bf16 tensors to fp16 (T4 compatibility)")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Mistral has no pad token → reuse eos_token (see docstring above)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Required for LoRA training

    logger.info(
        f"Model loaded: {model.num_parameters() / 1e9:.2f} B parameters "
        f"(quantized to 4-bit)"
    )
    logger.info(
        f"Tokenizer: vocab_size={tokenizer.vocab_size}, "
        f"pad_token='{tokenizer.pad_token}', padding_side='{tokenizer.padding_side}'"
    )

    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA Setup
# ---------------------------------------------------------------------------
def setup_lora(
    model: AutoModelForCausalLM,
    config: Dict[str, Any],
) -> AutoModelForCausalLM:
    """
    Prepare the quantized model for LoRA training and apply LoRA adapters.

    This is a two-step process:
      1. prepare_model_for_kbit_training: Fixes gradient computation for
         quantized (frozen) layers.  Without this, gradients flow incorrectly
         through the 4-bit weights and training diverges.
      2. get_peft_model: Wraps the model with trainable LoRA adapters
         injected into the target modules.

    WHY these LoRA hyperparameters:
      - r=16: Rank of the low-rank matrices.  r=8 underfits on instruction-
        following; r=32+ risks OOM on T4.  r=16 is the community standard
        for 7 B models on consumer GPUs.
      - alpha=32 (2×r): Scaling factor (alpha/r) controls update magnitude.
        The 2× heuristic comes from the original LoRA paper (Hu et al., 2021).
      - dropout=0.05: Light regularisation for a ~34 K row dataset.
        0.0 = no regularisation (overfits), 0.1 = too aggressive.
      - target_modules: All 7 linear layers (q/k/v/o_proj + gate/up/down_proj).
        Empirical work in 2024–2025 shows targeting MLP layers in addition to
        attention dramatically improves instruction-following quality, with
        minimal extra VRAM under QLoRA.

    Args:
        model: Quantized base model.
        config: Project configuration dictionary.

    Returns:
        PEFT-wrapped model with trainable LoRA adapters.
    """
    # Step 1: Fix gradient handling for quantized layers
    # This casts layer norms to fp32 and freezes the base model, then enables
    # gradient computation only for LoRA parameters.
    model = prepare_model_for_kbit_training(model)
    logger.info("Model prepared for k-bit training (layer norms → fp32, base frozen)")

    # Step 2: Define LoRA configuration
    lora_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get(
            "lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"],
        ),
        bias="none",  # LoRA paper recommends no bias adaptation
        task_type=TaskType.CAUSAL_LM,
    )

    # Step 3: Wrap model with LoRA adapters
    model = get_peft_model(model, lora_config)

    # Report trainable parameter count
    _print_trainable_parameters(model)

    return model


def _print_trainable_parameters(model: AutoModelForCausalLM) -> None:
    """
    Log the number and percentage of trainable parameters.

    WHY this matters:
      - Sanity check that LoRA is actually applied.  If trainable % is 100 %
        something is wrong (full fine-tuning).  If it's 0 % the model is
        fully frozen and won't learn.  Expect ~0.3 % for r=16 on 7 B params.
      - Useful for estimating memory usage: only trainable params need
        optimizer states (2× param size for AdamW).

    Args:
        model: The PEFT-wrapped model.
    """
    total_params = 0
    trainable_params = 0
    for _, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    trainable_pct = 100 * trainable_params / total_params
    logger.info(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({trainable_pct:.2f}%)"
    )
    logger.info(
        f"  → Trainable: {trainable_params / 1e6:.1f} M params "
        f"(~{trainable_params * 2 / 1e9:.2f} GB in fp16 for optimizer states)"
    )


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------
def load_processed_dataset(data_dir: str = "data/processed") -> DatasetDict:
    """
    Load the pre-processed dataset saved by data_prep.py.

    WHY load from disk (not re-process):
      - data_prep.py already cleaned, formatted, and split the data.
        Re-processing ~34 K rows wastes 2–5 minutes every training run.
      - Arrow format is memory-mapped: loading is nearly instant regardless
        of dataset size.
      - Decouples data processing from training — you can iterate on one
        without touching the other.

    Args:
        data_dir: Path to the directory created by data_prep.py.

    Returns:
        DatasetDict with 'train', 'validation', and 'test' splits.

    Raises:
        FileNotFoundError: If the processed data directory doesn't exist.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Processed data not found at '{data_dir}'.\n"
            f"Run `python data_prep.py` first to create it."
        )

    logger.info(f"Loading processed dataset from {data_dir}")
    dataset = DatasetDict.load_from_disk(data_dir)

    for split_name, split_data in dataset.items():
        logger.info(f"  {split_name}: {len(split_data):,} examples")

    # Verify the expected 'text' column exists
    if "text" not in dataset["train"].column_names:
        raise ValueError(
            f"Expected 'text' column in dataset but found: "
            f"{dataset['train'].column_names}. "
            f"Re-run data_prep.py to regenerate the processed data."
        )

    return dataset


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def create_training_args(
    config: Dict[str, Any],
    dry_run: bool = False,
) -> TrainingArguments:
    """
    Build TrainingArguments from the config file.

    WHY every setting is loaded from config (not hardcoded):
      - Single source of truth.  Changing a value in config.yaml is reflected
        everywhere: training, evaluation, logging.
      - Experiment tracking: the config file IS the experiment record.
        Commit it to git alongside your results for full reproducibility.

    T4-SPECIFIC CONSTRAINTS enforced here:
      - fp16=False, bf16=False: Disables PyTorch's GradScaler entirely.
        WHY: Mistral 7B weights are stored in bf16 on HuggingFace. Even after
        casting to fp16, some internal ops produce bf16 gradients that crash
        the GradScaler. Since QLoRA already compresses the model to ~4 GB,
        mixed-precision training is redundant for memory savings.
      - gradient_checkpointing=True: Essential on 16 GB VRAM.
      - optim="paged_adamw_32bit": Offloads optimizer states to CPU during
        memory spikes, preventing OOM during gradient accumulation.
      - per_device_train_batch_size=1: A single forward pass of Mistral 7B
        with 512 tokens uses ~10–12 GB.  Batch size > 1 → OOM.

    Args:
        config: Project configuration dictionary.
        dry_run: If True, override to train for only 3 steps.

    Returns:
        Configured TrainingArguments.
    """
    output_dir = config.get("output_dir", "./results")
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,

        # --- Epochs & Steps ---
        num_train_epochs=config.get("num_train_epochs", 1),

        # --- Batch Size ---
        # T4 constraint: batch_size=1 is the maximum that fits in 16 GB VRAM
        per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=config.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),

        # --- Optimiser ---
        # paged_adamw_32bit: Offloads optimizer states to CPU on memory spikes.
        # 32-bit because T4 doesn't support bf16 optimizer states.
        optim=config.get("optim", "paged_adamw_32bit"),
        learning_rate=config.get("learning_rate", 2e-4),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        weight_decay=config.get("weight_decay", 0.01),
        max_grad_norm=config.get("max_grad_norm", 0.3),

        # --- Precision ---
        # CRITICAL: Both set to False to disable PyTorch's GradScaler.
        # Mistral 7B has bf16 weights that crash the GradScaler on T4.
        # QLoRA already handles memory — mixed precision is not needed.
        fp16=False,
        bf16=False,

        # --- Memory Optimisation ---
        # Gradient checkpointing: ~20% slower but ~40% less VRAM.
        # ESSENTIAL on T4 — without it, backward pass OOMs.
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # --- Logging & Saving ---
        logging_steps=config.get("logging_steps", 10),
        logging_first_step=True,
        save_strategy=config.get("save_strategy", "epoch"),
        report_to="none",  # Disable W&B / TensorBoard for simplicity

        # --- Reproducibility ---
        seed=config.get("seed", 42),
        data_seed=config.get("seed", 42),

        # --- Misc ---
        remove_unused_columns=False,  # SFTTrainer handles column management
    )

    # Dry-run override: train for only 3 steps (for quick pipeline testing)
    if dry_run:
        training_args.max_steps = 3
        training_args.logging_steps = 1
        training_args.save_strategy = "no"
        training_args.num_train_epochs = 1
        logger.info("🧪 DRY RUN MODE: training for 3 steps only")

    return training_args


def create_trainer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    training_args: TrainingArguments,
    dataset: DatasetDict,
    config: Dict[str, Any],
) -> SFTTrainer:
    """
    Initialise the SFTTrainer (Supervised Fine-Tuning Trainer from trl).

    WHY SFTTrainer over plain Trainer:
      - Automatically handles tokenisation of the 'text' column, applying
        the causal-LM loss (predict next token), and optionally packing
        multiple short sequences into one max_seq_length window.
      - Removes ~50 lines of boilerplate (custom data collator, tokenisation
        inside a map() call, etc.).
      - Actively maintained by HuggingFace — benefits from upstream fixes.

    WHY dataset_text_field="text":
      - data_prep.py stores the fully formatted prompt (question + answer)
        in a column called "text".  SFTTrainer tokenises this column and
        computes the causal-LM loss over all tokens.

    WHY packing=False by default:
      - Packing concatenates short examples to fill max_seq_length, improving
        GPU utilisation.  But it can blend unrelated Q&A pairs, which hurts
        tasks where boundaries between examples matter.
      - Enable in config.yaml if training is too slow.

    Args:
        model: PEFT-wrapped model with LoRA adapters.
        tokenizer: Configured tokenizer.
        training_args: TrainingArguments instance.
        dataset: DatasetDict with at least a 'train' split.
        config: Project configuration dictionary.

    Returns:
        Configured SFTTrainer ready for .train().
    """
    # WHY SFTConfig instead of TrainingArguments:
    #   - trl v0.14+ unified training config into SFTConfig (extends TrainingArguments).
    #   - SFTConfig includes dataset_text_field, max_seq_length, packing directly.
    #   - We construct it explicitly with every parameter to avoid silent defaults.
    try:
        from trl import SFTConfig

        sft_config = SFTConfig(
            output_dir=training_args.output_dir,
            num_train_epochs=training_args.num_train_epochs,
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            learning_rate=training_args.learning_rate,
            lr_scheduler_type=training_args.lr_scheduler_type,
            warmup_ratio=training_args.warmup_ratio,
            weight_decay=training_args.weight_decay,
            max_grad_norm=training_args.max_grad_norm,
            optim=training_args.optim,
            # CRITICAL: Both False — disables GradScaler completely.
            fp16=False,
            bf16=False,
            gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=training_args.logging_steps,
            logging_first_step=True,
            save_strategy=training_args.save_strategy,
            report_to="none",
            seed=training_args.seed,
            remove_unused_columns=False,
            # SFTConfig-specific fields
            dataset_text_field="text",
            max_seq_length=config.get("max_seq_length", 512),
            packing=config.get("packing", False),
        )

        # Apply dry-run overrides if they were set
        if training_args.max_steps > 0:
            sft_config.max_steps = training_args.max_steps
            sft_config.logging_steps = 1
            sft_config.save_strategy = "no"

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=sft_config,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
        )
    except (ImportError, TypeError):
        # Fallback for older trl versions
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
        )

    # Estimate training time
    _log_training_estimate(dataset, config, training_args)

    return trainer


def _log_training_estimate(
    dataset: DatasetDict,
    config: Dict[str, Any],
    training_args: TrainingArguments,
) -> None:
    """
    Log estimated training duration based on dataset size and batch settings.

    WHY estimate up front:
      - So you know whether to grab a coffee (30 min) or go to sleep (8 hr)
        before the run finishes.
      - Helps catch config mistakes: if the estimate is 200 hours, something
        is probably wrong.

    Rough heuristic for T4 + QLoRA + Mistral 7B:
      ~1.5 seconds per training step (batch_size=1, seq_len=512).
    """
    num_train_examples = len(dataset["train"])
    effective_batch_size = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
    )
    num_epochs = training_args.num_train_epochs
    total_steps = int(num_train_examples / effective_batch_size * num_epochs)

    # Override if max_steps is set (e.g., dry run)
    if training_args.max_steps > 0:
        total_steps = training_args.max_steps

    # ~1.5 s/step is a rough T4 + QLoRA + Mistral-7B estimate
    estimated_seconds = total_steps * 1.5
    estimated_minutes = estimated_seconds / 60
    estimated_hours = estimated_minutes / 60

    logger.info("=" * 60)
    logger.info("TRAINING ESTIMATE")
    logger.info("=" * 60)
    logger.info(f"  Training examples : {num_train_examples:,}")
    logger.info(f"  Effective batch   : {effective_batch_size}")
    logger.info(f"  Epochs            : {num_epochs}")
    logger.info(f"  Total steps       : {total_steps:,}")
    if estimated_hours >= 1:
        logger.info(f"  Estimated time    : ~{estimated_hours:.1f} hours")
    else:
        logger.info(f"  Estimated time    : ~{estimated_minutes:.0f} minutes")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Training Log Export
# ---------------------------------------------------------------------------
def save_training_logs(
    trainer: SFTTrainer,
    output_dir: str,
) -> str:
    """
    Save training loss history as a JSON file for downstream plotting.

    WHY JSON (not CSV or pickle):
      - Human-readable and trivially parsed by plot_results.py.
      - Works across Python versions with no dependencies.
      - Easy to inspect manually or pipe into jq on the command line.

    The log contains one entry per logging step with:
      - step: global training step number
      - loss: training loss at that step
      - learning_rate: current LR (useful for debugging LR schedule)
      - epoch: fractional epoch number

    Args:
        trainer: The trainer after .train() has completed.
        output_dir: Directory to write the log file.

    Returns:
        Path to the saved JSON log file.
    """
    log_history = trainer.state.log_history

    # Extract only the training loss entries (exclude eval entries)
    training_logs = []
    for entry in log_history:
        if "loss" in entry:
            training_logs.append({
                "step": entry.get("step"),
                "loss": entry.get("loss"),
                "learning_rate": entry.get("learning_rate"),
                "epoch": entry.get("epoch"),
            })

    log_path = os.path.join(output_dir, "training_logs.json")
    with open(log_path, "w") as f:
        json.dump(training_logs, f, indent=2)

    logger.info(f"Training logs saved to {log_path} ({len(training_logs)} entries)")
    return log_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Full QLoRA fine-tuning pipeline:
      1. Load config
      2. Create quantization config (4-bit NF4, fp16 compute on T4)
      3. Load & quantize Mistral-7B-v0.3
      4. Apply LoRA adapters (r=16, all linear layers)
      5. Load processed dataset from data/processed/
      6. Train with SFTTrainer
      7. Save adapter weights + tokenizer + training logs

    The pipeline is designed to run end-to-end on a single Google Colab T4
    (16 GB VRAM, ~12 GB available after CUDA overhead).
    """
    # ----- CLI Arguments -----
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning of Mistral-7B-v0.3 for Medical Q&A"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Path to processed dataset directory (default: data/processed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Train for only 3 steps (for pipeline testing)",
    )
    args = parser.parse_args()

    # ----- Step 1: Load Config -----
    config = load_config(args.config)

    # ----- Step 2: Quantization Config -----
    bnb_config = create_bnb_config(config)

    # ----- Step 3: Load Model & Tokenizer -----
    model, tokenizer = load_model_and_tokenizer(config, bnb_config)

    # ----- Step 4: Apply LoRA -----
    model = setup_lora(model, config)

    # ----- Step 5: Load Dataset -----
    dataset = load_processed_dataset(args.data_dir)

    # ----- Step 6: Create Trainer -----
    training_args = create_training_args(config, dry_run=args.dry_run)
    trainer = create_trainer(model, tokenizer, training_args, dataset, config)

    # ----- Step 7: Train -----
    logger.info("🚀 Starting training...")
    start_time = time.time()

    try:
        train_result = trainer.train()
    except torch.cuda.OutOfMemoryError:
        logger.error(
            "💥 CUDA Out of Memory! Possible fixes:\n"
            "  1. Reduce max_seq_length in config.yaml (try 256)\n"
            "  2. Ensure gradient_checkpointing is True\n"
            "  3. Ensure per_device_train_batch_size is 1\n"
            "  4. Restart the Colab runtime to free leaked VRAM"
        )
        sys.exit(1)

    elapsed = time.time() - start_time
    elapsed_min = elapsed / 60

    logger.info(f"✅ Training complete in {elapsed_min:.1f} minutes")
    logger.info(f"   Final training loss: {train_result.training_loss:.4f}")

    # ----- Step 8: Save Adapter Weights & Tokenizer -----
    output_dir = config.get("output_dir", "./results")
    adapter_dir = os.path.join(output_dir, "final_adapter")
    os.makedirs(adapter_dir, exist_ok=True)

    # Save only the LoRA adapter weights (not the full 7 B model).
    # This produces a ~50–80 MB directory instead of ~14 GB.
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info(f"Adapter weights saved to {adapter_dir}")

    # ----- Step 9: Save Training Logs -----
    log_path = save_training_logs(trainer, output_dir)
    logger.info(f"Training logs (for plot_results.py) saved to {log_path}")

    # ----- Summary -----
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Model         : {config['model_name']}")
    logger.info(f"  LoRA rank     : {config.get('lora_r', 16)}")
    logger.info(f"  Training loss : {train_result.training_loss:.4f}")
    logger.info(f"  Duration      : {elapsed_min:.1f} minutes")
    logger.info(f"  Adapter saved : {adapter_dir}")
    logger.info(f"  Logs saved    : {log_path}")
    logger.info("=" * 60)
    logger.info("\n🎉 Next steps:")
    logger.info("  1. python evaluate.py    — Run evaluation metrics")
    logger.info("  2. python plot_results.py — Plot training curves")
    logger.info("  3. python merge_model.py  — Merge adapter into base model")


if __name__ == "__main__":
    main()
