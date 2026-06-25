---
title: Medical Q&A Assistant
emoji: 🏥
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
license: apache-2.0
tags:
  - medical
  - question-answering
  - fine-tuning
  - qlora
  - mistral
  - lora
  - peft
---

<div align="center">

# 🏥 Medical Q&A LLM Fine-Tuning with QLoRA

### Fine-tuning Mistral 7B for Medical Question Answering using Parameter-Efficient Adaptation

[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20Demo-HuggingFace%20Spaces-yellow)](https://huggingface.co/spaces/YOUR_USERNAME/medical-qa-mistral)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Fine-tuned on](https://img.shields.io/badge/Dataset-MedAlpaca%20Flashcards-orange)](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards)
[![Base Model](https://img.shields.io/badge/Base-Mistral%207B%20v0.3-purple)](https://huggingface.co/mistralai/Mistral-7B-v0.3)

</div>

---

> ⚠️ **Disclaimer**: This is a portfolio/educational project demonstrating LLM fine-tuning techniques. It is **NOT** a medical device and should **NOT** be used for medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals.

---

## 📋 Table of Contents

- [Problem Statement & Motivation](#-problem-statement--motivation)
- [Dataset Description](#-dataset-description)
- [Architecture Overview](#-architecture-overview)
- [Training Details](#-training-details)
- [Results](#-results)
- [Qualitative Examples](#-qualitative-examples)
- [What Didn't Work / Challenges](#-what-didnt-work--challenges)
- [Live Demo](#-live-demo)
- [How to Reproduce](#-how-to-reproduce)
- [Limitations & Ethical Considerations](#-limitations--ethical-considerations)
- [References](#-references)

---

## 🎯 Problem Statement & Motivation

**Problem**: General-purpose LLMs produce vague, sometimes inaccurate responses to medical questions because they lack domain-specific fine-tuning on curated medical knowledge.

**Goal**: Demonstrate that **parameter-efficient fine-tuning (LoRA/QLoRA)** can meaningfully improve a 7B-parameter model's medical Q&A performance — all within the constraints of free-tier hardware (Google Colab T4 GPU, 16GB VRAM).

**Why this matters for AI engineering**:
- **Efficiency**: QLoRA trains only ~0.3% of model parameters while matching full fine-tuning quality
- **Accessibility**: Production-quality fine-tuning on consumer hardware democratizes LLM customization
- **Real-world applicability**: Domain-specific LLMs are among the most requested capabilities in healthcare AI, legal tech, and enterprise applications

**Why Mistral 7B**:
- Open-weight model with a permissive license, suitable for research and deployment
- Strong baseline performance among 7B-class models
- Extensive community support and well-tested QLoRA compatibility
- 7B parameters is the largest model class that fits on a T4 GPU with 4-bit quantization

---

## 📊 Dataset Description

### Source: [MedAlpaca Medical Flashcards](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards)

Medical student flashcards rephrased into instruction-following format using GPT-3.5-Turbo, part of the peer-reviewed [MedAlpaca](https://arxiv.org/abs/2304.08247) project.

| Property | Value |
|----------|-------|
| **Total examples** | 33,955 |
| **Format** | Alpaca-style (instruction / input / output) |
| **Domain** | Clinical medicine, pharmacology, pathology, anatomy |
| **Source quality** | Medical student study materials |
| **License** | Open access |

### Preprocessing Pipeline

1. **Cleaning**: Removed rows with empty/null instruction or output fields, filtered answers shorter than 10 characters (likely noise like "N/A" or "Yes")
2. **Formatting**: Converted to instruction-style prompts with `### Question` / `### Answer` markers
3. **Context integration**: Merged the `instruction` and `input` fields (some examples have supplementary context in `input`)
4. **Splitting**: 85% train / 10% validation / 5% test (fixed seed=42 for reproducibility)

| Split | Examples | Purpose |
|-------|----------|---------|
| Train | ~28,800 | Model fine-tuning |
| Validation | ~3,400 | Overfitting detection |
| Test | ~1,700 | Final metric evaluation |

### Example (formatted prompt)

```
Below is a medical question. Provide a thorough and accurate answer.

### Question:
What are the common side effects of metformin in type 2 diabetes management?

### Answer:
Common side effects of metformin include gastrointestinal symptoms such as
nausea, vomiting, diarrhea, abdominal pain, and decreased appetite. These
effects are typically dose-dependent and often resolve with continued use.
The most serious but rare side effect is lactic acidosis, particularly in
patients with renal impairment...
```

---

## 🏗️ Architecture Overview

### Why LoRA (Low-Rank Adaptation)?

Full fine-tuning of Mistral 7B requires:
- **~14 GB** for model weights (fp16)
- **~28 GB** for optimizer states (AdamW stores 2 copies)
- **~14 GB** for gradients
- **Total: ~56+ GB** — far beyond T4's 16GB VRAM

**LoRA** (Hu et al., 2021) solves this by freezing all original weights and injecting small trainable **low-rank matrices** into each layer:

```
Original:    h = W · x           (W is frozen, no gradients needed)
With LoRA:   h = W · x + B·A·x  (only A and B are trained)

Where:
  W ∈ ℝ^(d×d)    — original weight matrix (e.g., 4096×4096)
  A ∈ ℝ^(r×d)    — LoRA down-projection (e.g., 16×4096)
  B ∈ ℝ^(d×r)    — LoRA up-projection (e.g., 4096×16)
  r = 16          — rank (the compression factor)
```

**QLoRA** (Dettmers et al., 2023) adds 4-bit quantization of the frozen weights, reducing VRAM from 14GB to ~4GB:

```
┌─────────────────────────────────────────────────────────┐
│                    Mistral 7B v0.3                      │
│                                                         │
│  ┌──────────────────┐    ┌──────────────────┐           │
│  │  Frozen Weights   │    │  LoRA Adapters    │          │
│  │  (4-bit NF4)      │    │  (fp16, trainable)│          │
│  │  ~4 GB VRAM       │    │  ~20M params      │          │
│  │                    │    │  ~40 MB VRAM      │          │
│  │  32 layers ×       │    │                    │         │
│  │  7 target modules  │    │  r=16, α=32       │         │
│  └──────────────────┘    └──────────────────┘           │
│                                                         │
│  Total VRAM: ~10-12 GB (fits on T4 16GB GPU)            │
└─────────────────────────────────────────────────────────┘
```

### LoRA Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Rank (r) | 16 | Standard for 7B models; higher ranks risk OOM on T4 |
| Alpha (α) | 32 | 2× rank — standard scaling heuristic from LoRA paper |
| Dropout | 0.05 | Light regularization for ~34K training examples |
| Target modules | All 7 linear layers | 2024-25 consensus: targeting MLP layers too (not just attention q/v) dramatically improves instruction-following |

**Target modules** (per transformer layer):
- Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- MLP: `gate_proj`, `up_proj`, `down_proj`

### Quantization Configuration (BitsAndBytes)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Quantization | 4-bit | Fits 7B model in ~4GB VRAM |
| Quant type | NF4 (NormalFloat4) | Optimal for normally-distributed neural network weights |
| Double quantization | Enabled | Quantizes quantization constants, saves ~0.4GB free |
| Compute dtype | float16 | **T4 has no native bf16 support** (Turing architecture) |

---

## 🏋️ Training Details

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 1 | LoRA adapters converge fast; more epochs risk overfitting |
| Batch size | 1 | Maximum that fits in T4 VRAM with Mistral 7B |
| Gradient accumulation | 4 steps | Effective batch size = 4 |
| Learning rate | 2×10⁻⁴ | QLoRA standard (higher than full FT due to fewer trainable params) |
| LR scheduler | Cosine | Gradual decay; outperforms linear for single-epoch runs |
| Warmup | 3% of steps | Prevents early gradient explosions |
| Max gradient norm | 0.3 | Aggressive clipping for 4-bit stability |
| Weight decay | 0.01 | Light L2 regularization |
| Optimizer | Paged AdamW 32-bit | Offloads optimizer states to CPU during memory spikes |
| Max sequence length | 512 tokens | 95%+ of examples fit; longer sequences risk OOM |
| Gradient checkpointing | ✅ Enabled | Trades ~20% speed for ~40% VRAM savings |
| Mixed precision | fp16 | T4 native format (bf16 not supported on Turing GPUs) |

### Hardware

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA Tesla T4 (16GB VRAM, Turing architecture) |
| Platform | Google Colab (free tier) |
| System RAM | ~12.7 GB |
| CUDA | 12.x |
| Training time | ~2-3 hours (estimated) |

### Trainable Parameters

| Category | Count | Percentage |
|----------|-------|------------|
| Total parameters | ~7.24B | 100% |
| Trainable (LoRA) | ~20M | **~0.28%** |
| Frozen (base model) | ~7.22B | 99.72% |

---

## 📈 Results

### Quantitative Metrics

> **Note**: Replace these placeholder values with your actual results after training.

| Metric | Baseline (Pre-FT) | Fine-Tuned | Improvement |
|--------|-------------------|------------|-------------|
| **ROUGE-L** | 0.XXX | 0.XXX | +XX.X% |
| **BLEU** | 0.XXX | 0.XXX | +XX.X% |
| **Perplexity** | X.XX | X.XX | -XX.X% (lower = better) |

### Training Loss Curve

> **TODO**: After training, run `python plot_results.py` and embed the generated plot:
>
> `![Training Loss Curve](results/training_loss.png)`

### Metric Comparison

> **TODO**: After evaluation, the bar chart will be at:
>
> `![Metric Comparison](results/metrics_comparison.png)`

---

## 💬 Qualitative Examples

> **TODO**: After running `python evaluate.py`, replace these with actual model outputs from `examples/comparison_examples.md`.

### Example 1

**Question**: What are the first-line treatments for hypertension?

| | Response |
|---|---------|
| **Baseline** | *(Pre-fine-tuning response — often generic or off-topic)* |
| **Fine-Tuned** | *(Post-fine-tuning response — should be specific and medically accurate)* |
| **Reference** | *(Ground truth from the dataset)* |

### Example 2

**Question**: Explain the mechanism of action of SSRIs in treating depression.

| | Response |
|---|---------|
| **Baseline** | *...* |
| **Fine-Tuned** | *...* |
| **Reference** | *...* |

*See `examples/comparison_examples.md` for all 10 side-by-side comparisons.*

---

## 🚧 What Didn't Work / Challenges

> **TODO**: Fill in this section after completing the project. Suggested areas to discuss:

- [ ] **Learning rate sensitivity**: Did you need to adjust from the default 2e-4?
- [ ] **Sequence length tradeoffs**: Did truncation at 512 tokens affect answer quality?
- [ ] **Overfitting signals**: Did validation loss increase after N steps?
- [ ] **Generation quality**: Any hallucination patterns? Repetitive outputs?
- [ ] **Memory management**: Any OOM crashes? What was your peak VRAM usage?
- [ ] **Evaluation challenges**: Are ROUGE/BLEU appropriate for open-ended medical Q&A?
- [ ] **Dataset quality**: Any issues with the MedAlpaca flashcard data?

*Discussing what didn't work demonstrates deeper understanding than just showing what did work.*

---

## 🚀 Live Demo

> **TODO**: Deploy to HuggingFace Spaces and update this link.

🔗 **[Try the Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/Krishna27s/medical-qa-mistral)**

The demo runs on [HuggingFace Spaces](https://huggingface.co/spaces) with CPU inference (slower but free).

---

## 🔬 How to Reproduce

### Prerequisites

- Google Colab account (free tier with T4 GPU)
- HuggingFace account (for dataset access and optional model hosting)
- ~3-4 hours total (data prep + training + evaluation)

### Step-by-Step

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/medical-qa-llm.git
cd medical-qa-llm

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare the dataset
#    Downloads from HuggingFace, cleans, formats, splits, saves to data/processed/
python data_prep.py

# 4. Fine-tune the model (runs on GPU — use Colab T4)
#    Estimated time: 2-3 hours on T4
python train.py

# 5. Evaluate (baseline vs fine-tuned comparison)
#    Generates metrics table and qualitative examples
python evaluate.py

# 6. Generate plots
python plot_results.py

# 7. Merge LoRA adapter into base model (for deployment)
python merge_and_export.py

# 8. Launch the Gradio app locally
python app.py
```

### Quick Test (Dry Run)

```bash
# Run a 3-step training to verify everything works
python train.py --dry-run
```

### Project Structure

```
medical-qa-llm/
├── config.yaml              # All hyperparameters (documented with WHY)
├── data_prep.py             # Data loading, cleaning, formatting, splitting
├── train.py                 # QLoRA fine-tuning with SFTTrainer
├── evaluate.py              # ROUGE-L, BLEU, perplexity comparison
├── merge_and_export.py      # Merge LoRA adapter into base model
├── plot_results.py          # Training loss curves + metric bar charts
├── app.py                   # Gradio inference UI for HF Spaces
├── requirements.txt         # Pinned dependencies
├── README.md                # This file
├── .gitignore               # Ignore checkpoints, cache, models
├── results/                 # Training outputs, metrics, plots
│   ├── training_log.json
│   ├── training_loss.png
│   ├── metrics_comparison.png
│   ├── metrics_comparison.md
│   └── eval_results.json
└── examples/
    └── comparison_examples.md   # Side-by-side qualitative analysis
```

---

## ⚖️ Limitations & Ethical Considerations

### Technical Limitations

- **Not a medical device**: This model has not been validated against clinical benchmarks (USMLE, MedQA-MCQA) or reviewed by medical professionals
- **Training data bias**: MedAlpaca flashcards are English-only, primarily covering Western medicine curricula
- **Hallucination risk**: LLMs can generate plausible-sounding but factually incorrect medical information
- **Evaluation metrics**: ROUGE-L and BLEU measure surface-level text overlap, not medical accuracy. A clinically meaningful evaluation would require expert human review
- **Limited context**: 512-token context window limits the model's ability to process long clinical scenarios

### Ethical Considerations

- **Do not deploy in clinical settings** without rigorous validation, regulatory approval, and human oversight
- **Bias in medical knowledge**: The training data reflects the biases of its source material (medical student flashcards, primarily US/European medical curricula)
- This project demonstrates **AI engineering skills**, not clinical AI safety. The gap between "works on benchmarks" and "safe for patients" is enormous.

---

## 📚 References

1. **LoRA**: Hu, E. J., et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). *arXiv:2106.09685*
2. **QLoRA**: Dettmers, T., et al. (2023). [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314). *arXiv:2305.14314*
3. **Mistral 7B**: Jiang, A. Q., et al. (2023). [Mistral 7B](https://arxiv.org/abs/2310.06825). *arXiv:2310.06825*
4. **MedAlpaca**: Han, T., et al. (2023). [MedAlpaca — An Open-Source Collection of Medical Conversational AI Models](https://arxiv.org/abs/2304.08247). *arXiv:2304.08247*
5. **NF4 Quantization**: Dettmers, T., et al. (2023). [The case for 4-bit precision](https://arxiv.org/abs/2212.09720). *arXiv:2212.09720*
6. **PEFT Library**: [HuggingFace PEFT Documentation](https://huggingface.co/docs/peft)
7. **TRL Library**: [HuggingFace TRL Documentation](https://huggingface.co/docs/trl)

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

- **Code**: Apache 2.0
- **Base model (Mistral 7B)**: Apache 2.0
- **Dataset (MedAlpaca)**: Check [dataset card](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards) for specific terms

---

<div align="center">

**Built as a portfolio project for AI Engineer roles**

*If you found this useful, give it a ⭐ on GitHub!*

[Report Issues](https://github.com/YOUR_USERNAME/medical-qa-llm/issues) · [View Demo](https://huggingface.co/spaces/YOUR_USERNAME/medical-qa-mistral)

</div>
