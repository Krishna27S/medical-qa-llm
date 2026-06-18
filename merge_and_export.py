"""
merge_and_export.py — LoRA Adapter Merging & Export
=====================================================

This script merges a trained LoRA adapter back into the base model, producing
a single, standalone model that can be loaded without the PEFT library.

WHY MERGING IS NEEDED:
  After QLoRA training, your "model" is actually TWO things:
    1. The original frozen base model weights (~14GB in fp16)
    2. A tiny LoRA adapter (~50–100MB) containing the learned deltas

  At inference time, PEFT intercepts every forward pass and adds the LoRA
  deltas on-the-fly:  output = base_weight @ x + (lora_A @ lora_B) @ x

  This works but has downsides for deployment:
    - Requires the `peft` library as a dependency
    - Slightly slower inference (extra matrix multiplications every forward pass)
    - More complex loading code (must load base + adapter separately)
    - Harder to serve with tools like vLLM, TGI, or llama.cpp that expect
      a single model directory

  Merging eliminates all of this by baking the LoRA deltas directly into the
  base weights:  merged_weight = base_weight + (lora_A @ lora_B) * scaling

  The result is a single model that loads like any standard HuggingFace model.

THE TRADEOFF:
  - LoRA adapter only: ~50–100MB on disk, but needs PEFT + base model at runtime
  - Merged model: ~14GB on disk (full fp16), but zero extra dependencies
  - For production/portfolio demos, the merged model is almost always better:
    simpler code, fewer failure modes, compatible with all serving frameworks

MEMORY REQUIREMENTS:
  - This script loads the FULL fp16 model (~14GB) into system RAM (not GPU VRAM).
  - On Google Colab, this uses the ~12GB system RAM (free tier) or ~50GB (Pro).
  - WHY system RAM, not GPU: We use device_map="cpu" because merging is a
    one-time operation (simple matrix addition) that doesn't need GPU acceleration.
    Loading on CPU also avoids competing with any GPU processes.
  - If you have <14GB system RAM, this script WILL crash with OOM. Use Colab Pro
    or a machine with ≥16GB RAM.

WHY WE CAN'T MERGE IN 4-BIT:
  QLoRA quantizes weights to 4-bit NormalFloat for training to fit in GPU VRAM.
  But you CANNOT merge LoRA adapters into quantized weights because:
    1. 4-bit weights are compressed/packed — they aren't real floating point numbers
       you can do arithmetic on directly.
    2. Merging requires: new_weight = old_weight + delta. This addition needs
       both operands in a real floating point format (fp16 or fp32).
    3. If you tried to dequantize → merge → re-quantize, you'd introduce double
       quantization error, degrading model quality unpredictably.
  Therefore, we MUST load the base model in full fp16 precision for merging.

USAGE:
  python merge_and_export.py                              # Uses config.yaml defaults
  python merge_and_export.py --config my_config.yaml      # Custom config
  python merge_and_export.py --adapter-path ./my_adapter   # Override adapter location
  python merge_and_export.py --output-path ./my_merged     # Override output location
  python merge_and_export.py --push-to-hub                 # Push merged model to HF Hub

Author: [Your Name]
Date: June 2026
"""

import os
import sys
import argparse
import yaml
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

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
# Inference Template (must match data_prep.py exactly)
# ---------------------------------------------------------------------------
# WHY duplicate here instead of importing from data_prep.py:
#   - This script should be self-contained — you might run it on a different
#     machine that doesn't have data_prep.py in the same directory.
#   - The template is short enough that duplication is preferable to coupling.
INFERENCE_TEMPLATE = """Below is a medical question. Provide a thorough and accurate answer.

### Question:
{question}

### Answer:
"""


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    WHY centralized config:
      - Single source of truth for model name, output directories, etc.
      - Prevents mismatches (e.g., training with model A but merging with model B).

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary of configuration values.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is malformed.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Make sure config.yaml exists in the current directory, "
            f"or specify a path with --config."
        )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def validate_adapter_path(adapter_path: str) -> Path:
    """
    Validate that the LoRA adapter directory exists and contains expected files.

    WHY validate before loading:
      - Loading a non-existent adapter gives a cryptic HuggingFace error.
      - Checking for adapter_config.json confirms this is actually a PEFT adapter,
        not just a random directory.
      - Fail fast with a helpful message instead of crashing 5 minutes into model loading.

    Args:
        adapter_path: Path to the trained LoRA adapter directory.

    Returns:
        Validated Path object.

    Raises:
        FileNotFoundError: If adapter directory or required files don't exist.
    """
    path = Path(adapter_path)

    if not path.exists():
        raise FileNotFoundError(
            f"❌ Adapter directory not found: {path}\n\n"
            f"This usually means one of:\n"
            f"  1. Training hasn't been run yet (run train.py first)\n"
            f"  2. The adapter path is wrong (check output_dir in config.yaml)\n"
            f"  3. The training output was deleted or moved\n\n"
            f"Expected location: {path.resolve()}"
        )

    # Check for the PEFT adapter config file
    adapter_config = path / "adapter_config.json"
    if not adapter_config.exists():
        # Sometimes the adapter is in a subdirectory (e.g., results/checkpoint-1000/)
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        checkpoint_dirs = [d for d in subdirs if d.name.startswith("checkpoint-")]

        if checkpoint_dirs:
            # Use the latest checkpoint
            latest = sorted(checkpoint_dirs, key=lambda d: d.name)[-1]
            if (latest / "adapter_config.json").exists():
                logger.warning(
                    f"No adapter_config.json in {path}, "
                    f"but found one in {latest.name}. Using that instead."
                )
                return latest

        raise FileNotFoundError(
            f"❌ No adapter_config.json found in {path}\n\n"
            f"This directory doesn't appear to contain a PEFT/LoRA adapter.\n"
            f"Contents: {[f.name for f in path.iterdir()]}\n\n"
            f"Make sure you're pointing to the directory that contains "
            f"adapter_config.json and adapter_model.safetensors (or .bin)."
        )

    logger.info(f"✅ Adapter directory validated: {path}")

    # Log adapter size for reference
    adapter_size_mb = sum(
        f.stat().st_size for f in path.rglob("*") if f.is_file()
    ) / (1024 * 1024)
    logger.info(f"   Adapter size on disk: {adapter_size_mb:.1f} MB")

    return path


def load_base_model(
    model_name: str,
    device_map: str = "cpu",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the base model in full fp16 precision (NOT quantized).

    WHY fp16 and NOT 4-bit:
      - LoRA merging requires real floating point weights. 4-bit NF4 weights
        are packed integers that can't participate in float arithmetic directly.
      - We need: merged = base_weight + (lora_A @ lora_B) * scaling
        This operation requires both operands in fp16 (or fp32).
      - The merged model will also be saved in fp16, which is the standard
        format for deployment with vLLM, TGI, and other serving tools.

    WHY device_map="cpu":
      - Merging is a one-time matrix addition — it doesn't need GPU.
      - Loading a 7B model in fp16 needs ~14GB. On Colab, system RAM (12–50GB)
        is usually more available than GPU VRAM (16GB on T4, which may already
        be occupied by other processes).
      - On machines with >16GB VRAM and no other GPU processes, you can use
        device_map="auto" for slightly faster merging.

    Args:
        model_name: HuggingFace model identifier (e.g., "mistralai/Mistral-7B-v0.3").
        device_map: Where to load the model — "cpu" (safer) or "auto" (uses GPU if available).

    Returns:
        Tuple of (model, tokenizer).
    """
    logger.info(f"Loading base model: {model_name}")
    logger.info(f"  Device map: {device_map}")
    logger.info(f"  Precision: float16 (NOT quantized — required for merging)")
    logger.info(
        f"  ⚠️  This will use ~14GB of system RAM. "
        f"If you run out of memory, close other applications."
    )

    # Load tokenizer first (lightweight, ~1MB)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # WHY trust_remote_code=True:
    #   Some models (including Mistral v0.3) may have custom tokenizer code
    #   hosted on HuggingFace. This flag allows loading it. It's safe for
    #   well-known models from official repos.

    # Ensure pad token is set (same as in training)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("  Set pad_token = eos_token (same as during training)")

    # Load model in fp16 on CPU
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
            # WHY low_cpu_mem_usage=True:
            #   Loads the model using a memory-efficient method that doesn't create
            #   a full copy during initialization. Reduces peak RAM usage by ~50%.
            #   Essential when working with limited system RAM.
            low_cpu_mem_usage=True,
        )
    except torch.cuda.OutOfMemoryError:
        logger.warning(
            "GPU OOM — falling back to CPU loading. This is normal for merging."
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    param_count = sum(p.numel() for p in model.parameters())
    model_size_gb = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024**3)
    logger.info(
        f"  ✅ Base model loaded: {param_count / 1e9:.2f}B params, "
        f"{model_size_gb:.1f} GB in memory"
    )

    return model, tokenizer


def merge_adapter(
    model: AutoModelForCausalLM,
    adapter_path: Path,
) -> AutoModelForCausalLM:
    """
    Load the LoRA adapter and merge it into the base model.

    HOW MERGING WORKS:
      For each layer that has a LoRA adapter (e.g., q_proj, v_proj, etc.):
        1. PEFT stores two small matrices: lora_A (r × d_in) and lora_B (d_out × r)
        2. The effective delta is: delta = lora_B @ lora_A * (alpha / r)
        3. merge_and_unload() computes: base_weight += delta
        4. Then it removes all LoRA modules, leaving a standard model

      After this, the model is identical to a model that was fully fine-tuned
      (from the mathematical perspective of the forward pass), but it was
      trained using only a fraction of the parameters.

    Args:
        model: Base model loaded in fp16.
        adapter_path: Path to the trained LoRA adapter.

    Returns:
        Merged model with LoRA weights baked in.
    """
    logger.info(f"Loading LoRA adapter from: {adapter_path}")

    # PeftModel.from_pretrained attaches the adapter to the base model
    # WHY we load the adapter separately (not during model init):
    #   - The base model was loaded in fp16 without quantization.
    #   - The adapter was TRAINED on a 4-bit quantized model, but the adapter
    #     weights themselves are in fp16 — so they're compatible.
    model = PeftModel.from_pretrained(
        model,
        str(adapter_path),
        # WHY is_trainable=False:
        #   We're not going to train further — just merge and save.
        #   This skips setting up training-related buffers, saving memory.
        is_trainable=False,
    )

    logger.info("  Adapter loaded. Merging weights into base model...")

    # THE KEY OPERATION: merge LoRA deltas into base weights
    # After this call:
    #   - All lora_A and lora_B matrices are multiplied and added to base weights
    #   - All PEFT wrapper modules are removed
    #   - The model is a standard HuggingFace model (no PEFT dependency needed)
    model = model.merge_and_unload()

    logger.info("  ✅ Merge complete. Model is now a standalone HuggingFace model.")

    # Verify: the model should no longer have any PEFT modules
    peft_modules = [
        name for name, _ in model.named_modules()
        if "lora" in name.lower()
    ]
    if peft_modules:
        logger.warning(
            f"⚠️  Found {len(peft_modules)} residual LoRA modules after merging. "
            f"This shouldn't happen — the merge may be incomplete."
        )
    else:
        logger.info("  Verified: no residual LoRA modules found (merge is clean).")

    return model


def save_merged_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_path: str,
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
) -> None:
    """
    Save the merged model and tokenizer to disk (and optionally to HuggingFace Hub).

    WHY save both model AND tokenizer:
      - The tokenizer must travel with the model. If someone loads your merged model
        with a different tokenizer (or wrong version), they'll get garbled output.
      - Saving them together in one directory makes the model fully self-contained.

    WHAT GETS SAVED:
      - model.safetensors (or sharded): The merged fp16 weights (~14GB)
      - config.json: Model architecture config
      - tokenizer.json, tokenizer_config.json, special_tokens_map.json: Tokenizer files
      - generation_config.json: Default generation settings

    Args:
        model: Merged model.
        tokenizer: Tokenizer.
        output_path: Directory to save the merged model.
        push_to_hub: Whether to also upload to HuggingFace Hub.
        hub_model_id: HuggingFace Hub model ID (e.g., "username/model-name").
    """
    output_path = Path(output_path)
    logger.info(f"Saving merged model to: {output_path}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model
    # WHY safe_serialization=True:
    #   Saves in .safetensors format instead of .bin (PyTorch pickle).
    #   Safetensors is faster to load, safer (no arbitrary code execution),
    #   and is the new HuggingFace standard.
    model.save_pretrained(
        str(output_path),
        safe_serialization=True,
    )

    # Save tokenizer
    tokenizer.save_pretrained(str(output_path))

    # Report saved file sizes
    total_size_bytes = sum(
        f.stat().st_size for f in output_path.rglob("*") if f.is_file()
    )
    total_size_gb = total_size_bytes / (1024**3)
    logger.info(f"  ✅ Saved! Total size on disk: {total_size_gb:.2f} GB")

    # List saved files for verification
    logger.info("  Saved files:")
    for f in sorted(output_path.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024**2)
            logger.info(f"    {f.name}: {size_mb:.1f} MB")

    # Optionally push to HuggingFace Hub
    if push_to_hub:
        if not hub_model_id:
            logger.warning(
                "⚠️  push_to_hub is True but hub_model_id is empty. "
                "Set hub_model_id in config.yaml (e.g., 'your-username/medical-qa-mistral-7b')."
            )
            return

        logger.info(f"Pushing to HuggingFace Hub: {hub_model_id}")
        try:
            model.push_to_hub(hub_model_id, safe_serialization=True)
            tokenizer.push_to_hub(hub_model_id)
            logger.info(f"  ✅ Pushed to https://huggingface.co/{hub_model_id}")
        except Exception as e:
            logger.error(
                f"❌ Failed to push to Hub: {e}\n"
                f"Make sure you're logged in: run `huggingface-cli login` first."
            )


def verify_merged_model(
    output_path: str,
    test_question: str = "What are the common symptoms of type 2 diabetes?",
    max_new_tokens: int = 128,
) -> None:
    """
    Verification step: reload the saved model and run a test inference.

    WHY verify after saving:
      - Saving can silently fail (disk full, permissions, corrupted tensors).
      - If we can load the model back AND generate coherent text, we have high
        confidence the merge and save were successful.
      - This catches issues like: wrong tokenizer saved, missing config files,
        corrupted safetensors, etc.

    NOTE: This loads the model a second time, so it temporarily needs ~28GB RAM
    (original + reloaded). If memory is tight, skip verification with --skip-verify.

    Args:
        output_path: Path to the saved merged model.
        test_question: A medical question to test generation.
        max_new_tokens: Max tokens to generate in the test.
    """
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION: Reloading merged model for test inference")
    logger.info("=" * 60)

    try:
        # Load the merged model fresh from disk
        logger.info(f"Loading merged model from: {output_path}")
        tokenizer = AutoTokenizer.from_pretrained(output_path)
        model = AutoModelForCausalLM.from_pretrained(
            output_path,
            torch_dtype=torch.float16,
            device_map="cpu",  # Verification on CPU — no GPU needed
            low_cpu_mem_usage=True,
        )
        logger.info("  ✅ Model loaded successfully from disk.")

        # Verify model architecture — should have NO LoRA modules
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Total parameters: {total_params / 1e9:.2f}B")

        # Run test inference
        logger.info(f"\n  Test question: {test_question}")
        prompt = INFERENCE_TEMPLATE.format(question=test_question)

        inputs = tokenizer(prompt, return_tensors="pt")

        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        logger.info("  Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                # WHY pad_token_id=eos_token_id:
                #   Prevents a warning about missing pad token during generation.
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated tokens (exclude the prompt)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        logger.info(f"\n  Generated response:\n  {'-' * 40}")
        # Print response with indentation for readability
        for line in response.strip().split("\n"):
            logger.info(f"  {line}")
        logger.info(f"  {'-' * 40}")
        logger.info(f"  Generated {len(generated_ids)} tokens.")

        # Basic sanity checks on the response
        if len(response.strip()) < 10:
            logger.warning(
                "⚠️  Response is very short. The model may not have learned much, "
                "or the merge may have issues. Check training logs."
            )
        else:
            logger.info("  ✅ Verification passed! Model generates coherent text.")

        # Clean up to free memory
        del model, tokenizer, inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        logger.error(
            f"❌ Verification failed: {e}\n"
            f"The merged model may be corrupted. Try re-running the merge."
        )
        raise


def print_summary(
    adapter_path: str,
    output_path: str,
    adapter_size_mb: float,
    merged_size_gb: float,
) -> None:
    """
    Print a human-readable summary of the merge operation.

    WHY a summary:
      - Makes the script output scannable in Colab logs.
      - Clearly shows the size tradeoff between adapter and merged model.
    """
    logger.info("\n" + "=" * 60)
    logger.info("MERGE & EXPORT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Adapter source:   {adapter_path}")
    logger.info(f"  Merged output:    {output_path}")
    logger.info(f"  Adapter size:     {adapter_size_mb:.1f} MB")
    logger.info(f"  Merged model:     {merged_size_gb:.2f} GB")
    logger.info(f"  Size increase:    {merged_size_gb * 1024 / max(adapter_size_mb, 0.1):.0f}x")
    logger.info(f"")
    logger.info(f"  WHY the merged model is so much larger:")
    logger.info(f"    The adapter only contains the LoRA delta weights (~0.3% of params).")
    logger.info(f"    The merged model contains ALL weights with deltas baked in.")
    logger.info(f"    Tradeoff: larger on disk, but simpler to deploy (no PEFT needed).")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Main pipeline: validate → load base → load adapter → merge → save → verify.

    The pipeline is designed to fail fast: we validate the adapter directory
    before spending 5+ minutes downloading and loading the base model.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Merge a trained LoRA adapter into the base model for deployment. "
            "Produces a single model directory that works without the PEFT library."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python merge_and_export.py                         # Use config.yaml defaults\n"
            "  python merge_and_export.py --adapter-path ./results # Custom adapter path\n"
            "  python merge_and_export.py --push-to-hub            # Upload to HF Hub\n"
            "  python merge_and_export.py --skip-verify             # Skip verification step\n"
            "  python merge_and_export.py --device-map auto         # Use GPU if available\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help=(
            "Path to the trained LoRA adapter directory. "
            "Overrides output_dir from config.yaml. "
            "Should contain adapter_config.json and adapter_model.safetensors."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help=(
            "Path to save the merged model. "
            "Overrides merged_model_dir from config.yaml."
        ),
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        default=False,
        help="Push merged model to HuggingFace Hub (overrides config).",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="cpu",
        choices=["cpu", "auto"],
        help=(
            "Device map for loading the base model. "
            "'cpu' is safer and uses system RAM (~14GB needed). "
            "'auto' uses GPU if available (faster but needs 16GB+ VRAM free). "
            "Default: cpu"
        ),
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        default=False,
        help=(
            "Skip the verification step (reloading and test inference). "
            "Use this if you're low on RAM — verification loads the model twice."
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 0: Load configuration
    # ------------------------------------------------------------------
    config = load_config(args.config)

    # Resolve paths: CLI args override config values
    adapter_path = args.adapter_path or config.get("output_dir", "./results")
    output_path = args.output_path or config.get("merged_model_dir", "./merged_model")
    model_name = config["model_name"]
    push_to_hub = args.push_to_hub or config.get("push_to_hub", False)
    hub_model_id = config.get("hub_model_id", "")

    logger.info("=" * 60)
    logger.info("LoRA ADAPTER MERGING & EXPORT")
    logger.info("=" * 60)
    logger.info(f"  Base model:     {model_name}")
    logger.info(f"  Adapter path:   {adapter_path}")
    logger.info(f"  Output path:    {output_path}")
    logger.info(f"  Device map:     {args.device_map}")
    logger.info(f"  Push to Hub:    {push_to_hub}")
    logger.info(f"  Skip verify:    {args.skip_verify}")
    logger.info("")

    # ------------------------------------------------------------------
    # Step 1: Validate adapter path (fail fast before downloading model)
    # ------------------------------------------------------------------
    validated_adapter_path = validate_adapter_path(adapter_path)

    # Compute adapter size for the summary
    adapter_size_mb = sum(
        f.stat().st_size for f in validated_adapter_path.rglob("*") if f.is_file()
    ) / (1024 * 1024)

    # ------------------------------------------------------------------
    # Step 2: Load the base model in fp16 (NOT quantized)
    # ------------------------------------------------------------------
    logger.info("")
    model, tokenizer = load_base_model(
        model_name=model_name,
        device_map=args.device_map,
    )

    # ------------------------------------------------------------------
    # Step 3: Merge LoRA adapter into base model
    # ------------------------------------------------------------------
    logger.info("")
    model = merge_adapter(model, validated_adapter_path)

    # ------------------------------------------------------------------
    # Step 4: Save merged model + tokenizer
    # ------------------------------------------------------------------
    logger.info("")
    save_merged_model(
        model=model,
        tokenizer=tokenizer,
        output_path=output_path,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
    )

    # Compute merged model size for the summary
    merged_size_gb = sum(
        f.stat().st_size
        for f in Path(output_path).rglob("*")
        if f.is_file()
    ) / (1024**3)

    # Free the original model from memory before verification
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 5: Verify by reloading and running test inference
    # ------------------------------------------------------------------
    if not args.skip_verify:
        verify_merged_model(
            output_path=output_path,
            test_question="What are the common symptoms of type 2 diabetes?",
            max_new_tokens=config.get("eval_max_new_tokens", 128),
        )
    else:
        logger.info("\n⏩ Skipping verification (--skip-verify flag set).")

    # ------------------------------------------------------------------
    # Step 6: Print summary
    # ------------------------------------------------------------------
    print_summary(
        adapter_path=str(validated_adapter_path),
        output_path=output_path,
        adapter_size_mb=adapter_size_mb,
        merged_size_gb=merged_size_gb,
    )

    logger.info("\n✅ Merge & export complete!")
    logger.info(f"   Your merged model is ready at: {output_path}")
    logger.info(f"   To use it: AutoModelForCausalLM.from_pretrained('{output_path}')")
    logger.info(f"   No PEFT library needed! 🎉")


if __name__ == "__main__":
    main()
