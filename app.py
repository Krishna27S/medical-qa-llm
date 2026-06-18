"""
app.py — Gradio Deployment Interface for Medical Q&A LLM
========================================================

A clean, production-style Gradio Blocks interface for the fine-tuned
Mistral-7B medical Q&A model.

DESIGN DECISIONS:
  - Gradio Blocks (not Interface) for full layout control: disclaimer banner,
    sidebar accordion, and custom button placement.
  - Model loading is lazy (happens once at startup, cached globally) to avoid
    reloading on every request.
  - Supports three deployment targets:
      1. Local dev:   `python app.py` or `gradio app.py`
      2. Colab:       `demo.launch(share=True)` for a public URL
      3. HF Spaces:   Push repo with this file + README.md YAML header
  - 4-bit quantization on GPU keeps VRAM usage ~5GB, leaving room for KV cache.
  - Falls back gracefully to CPU (fp32) if no GPU — slower but functional.

HF SPACES README.md YAML (paste at the top of README.md for Spaces deployment):
---
title: Medical Q&A Assistant
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.34.2"
app_file: app.py
pinned: false
license: apache-2.0
tags:
  - medical
  - qlora
  - mistral
---

USAGE:
  python app.py                        # Launch locally (default port 7860)
  python app.py --config my.yaml       # Custom config
  python app.py --share                # Create public Gradio link
  python app.py --port 8080            # Custom port

Author: [Your Name]
Date: June 2026
"""

import os
import argparse
import logging
import yaml
from typing import Dict, Any, Optional, Tuple

import gradio as gr
import torch

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
# WHY duplicate the template here instead of importing from data_prep.py:
#   - app.py should be self-contained for HuggingFace Spaces deployment.
#   - Spaces clones your repo and runs app.py directly; extra imports can
#     break if module structure changes.
#   - The template is small enough that duplication is acceptable. If it
#     diverges, tests will catch it.

INFERENCE_TEMPLATE = """Below is a medical question. Provide a thorough and accurate answer.

### Question:
{question}

### Answer:
"""

# ---------------------------------------------------------------------------
# Medical Disclaimer
# ---------------------------------------------------------------------------
DISCLAIMER_HTML = """
<div style="
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    color: white;
    padding: 16px 20px;
    border-radius: 10px;
    margin-bottom: 16px;
    font-size: 15px;
    line-height: 1.5;
    box-shadow: 0 2px 8px rgba(238, 90, 36, 0.3);
">
    <strong>⚠️ DISCLAIMER:</strong> This is a portfolio/educational project
    demonstrating LLM fine-tuning techniques. It is <strong>NOT</strong> a
    medical device and should <strong>NOT</strong> be used for medical advice,
    diagnosis, or treatment. Always consult qualified healthcare professionals.
</div>
"""

# ---------------------------------------------------------------------------
# Example Questions
# ---------------------------------------------------------------------------
# WHY these five examples:
#   - They span different medical subdomains (anatomy, pharmacology,
#     pathophysiology, clinical, biochemistry) to showcase model breadth.
#   - They're phrased similarly to the training data (flashcard-style) so
#     the model produces its best answers.
#   - Short enough that users see a complete answer without scrolling.

EXAMPLE_QUESTIONS = [
    ["What are the main symptoms and risk factors for Type 2 Diabetes Mellitus?"],
    ["Explain the mechanism of action of beta-blockers in treating hypertension."],
    ["What is the difference between Crohn's disease and ulcerative colitis?"],
    ["Describe the role of the renin-angiotensin-aldosterone system in blood pressure regulation."],
    ["What are the common causes and treatment options for iron deficiency anemia?"],
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML, with sensible defaults if file is missing.

    WHY defaults instead of crashing:
      - On HuggingFace Spaces, config.yaml might not be present if the user
        only uploaded app.py and the model. Defaults keep the app functional.
      - Locally, the config is always available and takes precedence.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Configuration dictionary.
    """
    defaults = {
        "model_name": "mistralai/Mistral-7B-v0.3",
        "merged_model_dir": "./merged_model",
        "output_dir": "./results",
        "app_title": "🏥 Medical Q&A Assistant",
        "app_description": "Fine-tuned Mistral 7B for medical question answering",
        "app_max_new_tokens": 512,
        "app_temperature": 0.7,
        "app_top_p": 0.9,
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "float16",
        "lora_r": 16,
        "lora_alpha": 32,
    }

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            file_config = yaml.safe_load(f) or {}
        defaults.update(file_config)
        logger.info(f"Loaded config from {config_path}")
    else:
        logger.warning(
            f"Config file '{config_path}' not found — using built-in defaults. "
            f"This is expected on HuggingFace Spaces."
        )

    return defaults


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
# Global references — populated once by load_model_and_tokenizer().
_model = None
_tokenizer = None
_device = None


def get_device() -> str:
    """
    Auto-detect the best available device.

    WHY auto-detect instead of hardcoding:
      - The same app.py runs on Colab (T4 GPU), Spaces (variable GPU),
        and local laptops (CPU-only). Auto-detect makes it portable.

    Returns:
        "cuda" if a GPU is available, otherwise "cpu".
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        # Handle both old PyTorch (total_memory) and new PyTorch (total_mem)
        props = torch.cuda.get_device_properties(0)
        vram_bytes = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
        vram_gb = vram_bytes / 1e9
        logger.info(f"GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        return "cuda"
    else:
        logger.info("No GPU detected — running on CPU (inference will be slower)")
        return "cpu"


def load_model_and_tokenizer(
    config: Dict[str, Any],
) -> Tuple[Any, Any, str]:
    """
    Load the model and tokenizer, trying multiple strategies in order.

    Loading priority:
      1. Merged model (merged_model/) — fastest, no adapter merging needed
      2. Base model + LoRA adapter (results/) — requires PEFT to merge at runtime
      3. Base model only (fallback) — runs without fine-tuning, for demo purposes

    WHY this priority order:
      - The merged model is a single directory with the full fine-tuned weights;
        it loads like any HF model and doesn't need the PEFT library at all.
      - If only the adapter is available (e.g., you haven't run merge.py yet),
        we load the base model in quantized form and attach the adapter.
      - The base-model-only fallback ensures the app always launches, even if
        no training has been done yet — useful during development.

    GPU mode:
      - Uses 4-bit quantization (BitsAndBytes) to fit in 16GB VRAM.
      - compute_dtype=float16 because T4 has no native bf16 support.

    CPU mode:
      - Loads in float32 (BitsAndBytes quantization requires CUDA).
      - Slower but fully functional for testing and demos.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of (model, tokenizer, device_string).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = get_device()
    merged_dir = config["merged_model_dir"]
    adapter_dir = config["output_dir"]
    base_model_name = config["model_name"]

    model = None
    tokenizer = None

    # --- Strategy 1: Load merged model ---
    if os.path.isdir(merged_dir):
        logger.info(f"Loading merged model from {merged_dir}/...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                merged_dir, trust_remote_code=True
            )

            if device == "cuda":
                model = _load_quantized(merged_dir, config)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    merged_dir,
                    torch_dtype=torch.float16,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                )

            logger.info("✅ Merged model loaded successfully.")
            return model, tokenizer, device

        except Exception as e:
            logger.warning(f"Failed to load merged model: {e}")
            logger.info("Falling back to adapter loading...")

    # --- Strategy 2: Load base model + LoRA adapter ---
    adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        logger.info(f"Loading base model + adapter from {adapter_dir}/...")
        try:
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name, trust_remote_code=True
            )

            if device == "cuda":
                base_model = _load_quantized(base_model_name, config)
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                )

            model = PeftModel.from_pretrained(base_model, adapter_dir)
            logger.info("✅ Base model + adapter loaded successfully.")
            return model, tokenizer, device

        except ImportError:
            logger.error(
                "PEFT library not installed. Install with: pip install peft\n"
                "Or run merge.py first to create a merged model."
            )
        except Exception as e:
            logger.warning(f"Failed to load adapter: {e}")
            logger.info("Falling back to base model only...")

    # --- Strategy 3: Base model only (no fine-tuning) ---
    logger.warning(
        "⚠️  No fine-tuned model found. Loading the BASE model (un-fine-tuned). "
        "Answers will be lower quality. Run training first for best results."
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True
        )

        if device == "cuda":
            model = _load_quantized(base_model_name, config)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )

        logger.info("✅ Base model loaded (no fine-tuning applied).")
        return model, tokenizer, device

    except Exception as e:
        logger.error(f"❌ Failed to load any model: {e}")
        raise RuntimeError(
            f"Could not load model. Ensure '{base_model_name}' is accessible "
            f"and you have sufficient RAM/VRAM. Error: {e}"
        )


def _load_quantized(
    model_name_or_path: str, config: Dict[str, Any]
) -> Any:
    """
    Load a model with 4-bit BitsAndBytes quantization for GPU deployment.

    WHY a helper function:
      - The same quantization config is used for merged model, base+adapter,
        and base-only loading. DRY principle avoids triple duplication.

    WHY 4-bit for inference (not just training):
      - A 7B model in fp16 uses ~14GB VRAM, leaving little room for KV cache
        during generation. 4-bit uses ~4GB, giving plenty of headroom.
      - Quality loss from 4-bit inference is negligible for short-form Q&A.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        config: Configuration dictionary with quantization settings.

    Returns:
        Quantized model on GPU.
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    # Map string dtype to torch dtype
    compute_dtype_str = config.get("bnb_4bit_compute_dtype", "float16")
    compute_dtype = (
        torch.float16 if compute_dtype_str == "float16" else torch.bfloat16
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def generate_answer(
    question: str,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
) -> str:
    """
    Generate a medical answer for the given question.

    WHY we extract only the answer portion:
      - The model generates the full prompt echo + answer. Users only care
        about the answer, not the template scaffolding.
      - Splitting on '### Answer:' and taking the last segment handles both
        the case where the model echoes the prompt and where it doesn't.

    WHY temperature and top_p are user-controllable:
      - Temperature=0.1 gives deterministic, factual responses (good for
        medical Q&A). Temperature=0.9 gives more creative/varied responses.
      - Exposing these as sliders lets users explore the quality/diversity
        tradeoff — a great portfolio demo talking point.

    Args:
        question: The medical question to answer.
        temperature: Sampling temperature (higher = more random).
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        The generated answer text, cleaned of template artifacts.
    """
    global _model, _tokenizer, _device

    if _model is None or _tokenizer is None:
        return (
            "❌ Model not loaded. Please check the server logs for errors. "
            "Common causes:\n"
            "  • No model files in merged_model/ or results/\n"
            "  • Insufficient RAM/VRAM\n"
            "  • Missing dependencies (transformers, bitsandbytes, peft)"
        )

    if not question or not question.strip():
        return "Please enter a medical question."

    # Format the question using the inference template
    prompt = INFERENCE_TEMPLATE.format(question=question.strip())

    # Tokenize
    inputs = _tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(_model.device)
    attention_mask = inputs["attention_mask"].to(_model.device)

    # Generate
    try:
        with torch.no_grad():
            outputs = _model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=temperature > 0.01,  # Greedy if temperature ≈ 0
                repetition_penalty=1.15,
                pad_token_id=_tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (skip the prompt)
        generated_ids = outputs[0][input_ids.shape[-1]:]
        answer = _tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Clean up: remove any residual template markers
        # Sometimes the model generates another "### Question:" block
        answer = answer.split("### Question:")[0].strip()
        answer = answer.split("### Answer:")[-1].strip()

        if not answer:
            return (
                "The model generated an empty response. Try adjusting the "
                "temperature or rephrasing your question."
            )

        return answer

    except torch.cuda.OutOfMemoryError:
        # Clear VRAM and return a helpful error
        torch.cuda.empty_cache()
        return (
            "❌ GPU out of memory. Try reducing 'Max Tokens' to a lower value "
            "(e.g., 128) or restart the app."
        )
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        return f"❌ An error occurred during generation: {str(e)}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_interface(config: Dict[str, Any]) -> gr.Blocks:
    """
    Build the Gradio Blocks interface.

    WHY Blocks over Interface:
      - Blocks gives full layout control: we need a disclaimer banner on top,
        an accordion sidebar, sliders below the main I/O, and custom button
        placement. gr.Interface can't do this.
      - Blocks also supports custom CSS/HTML injection for the styled disclaimer.

    WHY gr.themes.Soft():
      - Clean, modern look with rounded corners and soft colors.
      - Good contrast for readability — important for a medical context.
      - Built-in dark mode support (Gradio handles this automatically).

    Args:
        config: Configuration dictionary.

    Returns:
        A gr.Blocks instance ready to launch.
    """
    app_title = config.get("app_title", "🏥 Medical Q&A Assistant")
    app_description = config.get(
        "app_description",
        "Fine-tuned Mistral 7B for medical question answering",
    )
    default_temp = config.get("app_temperature", 0.7)
    default_max_tokens = config.get("app_max_new_tokens", 256)

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="orange",
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(theme=theme) as demo:
        # --- Header ---
        gr.Markdown(f"# {app_title}")
        gr.Markdown(f"*{app_description}*")

        # --- Medical Disclaimer (prominent, styled) ---
        gr.HTML(DISCLAIMER_HTML)

        with gr.Row():
            # --- Main Column: Q&A Interface ---
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="Your Medical Question",
                    placeholder=(
                        "e.g., What are the common symptoms of hypothyroidism "
                        "and how is it diagnosed?"
                    ),
                    lines=3,
                    max_lines=6,
                )

                with gr.Row():
                    submit_btn = gr.Button(
                        "🔍 Submit",
                        variant="primary",
                        scale=2,
                    )
                    clear_btn = gr.Button(
                        "🗑️ Clear",
                        variant="secondary",
                        scale=1,
                    )

                answer_output = gr.Textbox(
                    label="Model's Answer",
                    lines=10,
                    max_lines=20,
                    interactive=False,
                )

                # --- Generation Parameter Sliders ---
                with gr.Accordion("⚙️ Generation Settings", open=False):
                    gr.Markdown(
                        "Adjust these to control answer style. Lower temperature "
                        "= more factual/deterministic. Higher = more creative/varied."
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=default_temp,
                        step=0.05,
                        label="Temperature",
                        info="0.1 = deterministic, 1.0 = creative",
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=64,
                        maximum=512,
                        value=default_max_tokens,
                        step=32,
                        label="Max Tokens",
                        info="Maximum length of the generated answer",
                    )

                # --- Example Questions ---
                gr.Examples(
                    examples=EXAMPLE_QUESTIONS,
                    inputs=question_input,
                    label="📋 Try These Example Questions",
                )

            # --- Sidebar Column: Model Info ---
            with gr.Column(scale=1, min_width=280):
                with gr.Accordion("ℹ️ Model Information", open=True):
                    gr.Markdown(
                        """
| Property | Value |
|---|---|
| **Base Model** | Mistral-7B-v0.3 |
| **Fine-tuning** | QLoRA (r=16, α=32) |
| **Dataset** | MedAlpaca Medical Flashcards (34K examples) |
| **Training** | 1 epoch on Google Colab T4 |
| **Quantization** | 4-bit NF4 (inference) |
"""
                    )

                with gr.Accordion("📊 Technical Details", open=False):
                    device_info = (
                        f"🟢 GPU: {torch.cuda.get_device_name(0)}"
                        if torch.cuda.is_available()
                        else "🟡 CPU mode (slower inference)"
                    )
                    gr.Markdown(
                        f"""
**Device:** {device_info}

**LoRA Config:**
- Rank: {config.get('lora_r', 16)}
- Alpha: {config.get('lora_alpha', 32)}
- Target modules: All linear layers

**Prompt Format:**
```
### Question:
<your question>

### Answer:
<model generates here>
```
"""
                    )

                with gr.Accordion("⚠️ Limitations", open=False):
                    gr.Markdown(
                        """
- Trained on **flashcard-style** medical Q&A — best for factual,
  textbook-style questions.
- May produce **incorrect or outdated** information.
- Does **not** have access to patient records, lab results,
  or real-time medical databases.
- **Not validated** for clinical use.
- Responses may vary with temperature settings.
"""
                    )

        # --- Event Handlers ---
        submit_btn.click(
            fn=generate_answer,
            inputs=[question_input, temperature_slider, max_tokens_slider],
            outputs=answer_output,
            show_progress="full",
        )

        # Also trigger on Enter key in the text box
        question_input.submit(
            fn=generate_answer,
            inputs=[question_input, temperature_slider, max_tokens_slider],
            outputs=answer_output,
            show_progress="full",
        )

        clear_btn.click(
            fn=lambda: ("", ""),
            inputs=None,
            outputs=[question_input, answer_output],
        )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Entry point: parse CLI args, load model, build UI, and launch.

    WHY argparse for a Gradio app:
      - `--share` creates a public URL (essential for Colab demos).
      - `--port` avoids conflicts when running multiple apps locally.
      - `--config` lets you swap configs without editing code.
      - On HF Spaces, none of these are needed — Spaces calls app.py directly.
    """
    parser = argparse.ArgumentParser(
        description="Launch the Medical Q&A Gradio interface"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link (useful for Colab)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to serve the app on (default: 7860)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load model and tokenizer (global singletons)
    global _model, _tokenizer, _device
    logger.info("=" * 60)
    logger.info("LOADING MODEL — this may take a few minutes...")
    logger.info("=" * 60)

    try:
        _model, _tokenizer, _device = load_model_and_tokenizer(config)
    except RuntimeError as e:
        logger.error(f"Model loading failed: {e}")
        logger.warning(
            "The app will launch but inference will return error messages. "
            "Fix the model loading issue and restart."
        )
        # _model and _tokenizer remain None; generate_answer() handles this

    # Build and launch the Gradio interface
    demo = build_interface(config)
    logger.info("=" * 60)
    logger.info(f"🚀 Launching Gradio app on port {args.port}...")
    logger.info("=" * 60)

    demo.launch(
        server_name="0.0.0.0",  # Accept connections from any IP (needed for Spaces/Colab)
        server_port=args.port,
        share=args.share,
        title=config.get("app_title", "Medical Q&A Assistant"),
    )


if __name__ == "__main__":
    main()
