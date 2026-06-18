"""
plot_results.py — Visualization for Medical Q&A Fine-Tuning Results
====================================================================

This script creates two publication-quality plots:
  1. Training Loss Curve — loss vs. training steps with EMA smoothing
  2. Metric Comparison Bar Chart — baseline vs. fine-tuned model evaluation

WHY a dedicated plotting script (not inline in train.py / evaluate.py):
  - Separation of concerns: training, evaluation, and visualization are
    independent stages. You shouldn't need to retrain just to tweak a plot.
  - Reproducibility: plots can be regenerated from saved JSON logs at any time.
  - Portfolio value: clean, publication-ready charts demonstrate data
    communication skills — a key ML engineering competency.

USAGE:
  python plot_results.py                         # Uses defaults
  python plot_results.py --config my.yaml        # Custom config
  python plot_results.py --results-dir ./run2     # Different results directory

Author: [Your Name]
Date: June 2026
"""

import os
import json
import argparse
import yaml
import logging
from typing import Dict, Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

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
# Global Style Configuration
# ---------------------------------------------------------------------------
# WHY custom styling instead of default matplotlib:
#   - Default matplotlib looks dated and amateurish for a portfolio project.
#   - We use a seaborn-inspired style with explicit overrides for fine control.
#   - DPI=150 balances file size and crispness for both screen and print.
#   - Minimum 12pt fonts ensure readability even when plots are embedded
#     in a README or shrunk in a slide deck.

STYLE_CONFIG = {
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.edgecolor": "#CCCCCC",
    "axes.grid": True,
    "axes.axisbelow": True,           # Grid lines behind data
    "grid.color": "#E0E0E0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#CCCCCC",
}

# Color palette — chosen for accessibility and print-friendliness.
# WHY these specific colors:
#   - Raw loss uses a light, muted tone so it doesn't dominate the plot.
#   - EMA line uses a bold blue that stands out as the "signal".
#   - Baseline gray vs. fine-tuned teal creates clear visual hierarchy:
#     the muted gray recedes, making the teal "pop" as the improved result.
COLORS = {
    "loss_raw": "#B0C4DE",       # Light steel blue — subtle background
    "loss_ema": "#1F4E79",       # Dark navy blue — strong signal line
    "baseline": "#9E9E9E",       # Medium gray — muted, "before"
    "finetuned": "#00897B",      # Teal — vibrant, "after"
    "perplexity_base": "#BDBDBD",  # Light gray for perplexity baseline
    "perplexity_ft": "#FF7043",    # Warm orange for perplexity (lower is better)
}


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    WHY load config even for plotting:
      - The results directory path is defined in config.yaml (output_dir).
      - Keeps all paths centralized — no hardcoded magic strings.
      - Allows different experiment runs to use different output directories.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary of configuration values.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def load_json_file(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Safely load a JSON file with helpful error messages.

    WHY graceful handling instead of crashing:
      - The user might run plot_results.py before training or evaluation is
        complete. Instead of a confusing traceback, we print a clear message
        explaining what's missing and how to fix it.
      - This makes the script usable as a "check status" tool during training.

    Args:
        filepath: Absolute or relative path to the JSON file.

    Returns:
        Parsed JSON as a dict/list, or None if the file doesn't exist.
    """
    if not os.path.exists(filepath):
        logger.warning(
            f"File not found: {filepath}\n"
            f"  → This file is created by another script. Run the appropriate\n"
            f"    pipeline step first:\n"
            f"      - training_log.json  → created by train.py\n"
            f"      - eval_results.json  → created by evaluate.py"
        )
        return None

    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded {filepath} ({os.path.getsize(filepath)} bytes)")
        return data
    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to parse {filepath}: {e}\n"
            f"  → The file may be incomplete (training still running?).\n"
            f"  → Try again after training completes."
        )
        return None


def compute_ema(values: List[float], alpha: float = 0.1) -> List[float]:
    """
    Compute Exponential Moving Average (EMA) for smoothing noisy loss curves.

    WHY EMA over simple moving average:
      - EMA gives more weight to recent values, so the smoothed curve tracks
        the trend more responsively. A simple moving average introduces lag.
      - alpha=0.1 means each new value contributes 10% and the history
        contributes 90%, giving heavy smoothing that reveals the trend
        without hiding genuine loss spikes.
      - EMA is the standard in TensorBoard and Weights & Biases.

    WHY alpha=0.1 (not 0.3 or 0.01):
      - 0.3 is too responsive — the smoothed line is almost as noisy as raw.
      - 0.01 is too sluggish — it takes hundreds of steps to react.
      - 0.1 is the sweet spot for ~7,000 training steps (29K rows / batch 4).

    Args:
        values: Raw loss values (one per logging step).
        alpha: Smoothing factor. Lower = smoother. Range: (0, 1).

    Returns:
        Smoothed values of the same length as input.
    """
    if not values:
        return []

    ema = [values[0]]  # Initialize with first value (no history yet)
    for v in values[1:]:
        ema.append(alpha * v + (1 - alpha) * ema[-1])
    return ema


def plot_training_loss(
    training_log: List[Dict[str, Any]],
    output_path: str,
    ema_alpha: float = 0.1,
) -> None:
    """
    Plot training loss vs. steps with a smoothed EMA trend line.

    WHY two lines (raw + smoothed):
      - Raw loss is noisy because each logged value is from a single batch
        (or small gradient accumulation group). Showing only raw data makes
        it hard to see the training trend.
      - The EMA line reveals the true trend: is loss decreasing? Plateauing?
        Diverging? This is what you actually care about.
      - Showing both lets viewers see the noise level AND the trend.

    Design choices:
      - Raw loss as thin, semi-transparent line: visible but doesn't dominate.
      - EMA as thick, opaque line: the hero of the plot.
      - Grid lines help estimate exact loss values at specific steps.

    Args:
        training_log: List of dicts, each with 'step' and 'loss' keys.
                      Produced by train.py during training.
        output_path: Where to save the PNG file.
        ema_alpha: Smoothing factor for the EMA trend line.
    """
    # Extract steps and losses from the training log
    steps = [entry["step"] for entry in training_log]
    losses = [entry["loss"] for entry in training_log]

    if not steps:
        logger.error("Training log is empty — nothing to plot.")
        return

    # Compute smoothed trend line
    ema_losses = compute_ema(losses, alpha=ema_alpha)

    logger.info(
        f"Plotting training loss: {len(steps)} data points, "
        f"steps {steps[0]}–{steps[-1]}, "
        f"loss range [{min(losses):.4f}, {max(losses):.4f}]"
    )

    # --- Create the figure ---
    with plt.rc_context(STYLE_CONFIG):
        fig, ax = plt.subplots(figsize=(10, 5))

        # Raw loss — thin, semi-transparent to show noise without cluttering
        ax.plot(
            steps,
            losses,
            color=COLORS["loss_raw"],
            linewidth=0.8,
            alpha=0.6,
            label="Raw Loss",
        )

        # Smoothed EMA — thick, prominent to show the trend
        ax.plot(
            steps,
            ema_losses,
            color=COLORS["loss_ema"],
            linewidth=2.5,
            label=f"Smoothed (EMA α={ema_alpha})",
        )

        # Annotations
        ax.set_title("Training Loss — Medical Q&A Fine-Tuning (QLoRA)")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right")

        # Format x-axis: use comma separators for large step counts (e.g., 7,250)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{int(x):,}"
        ))

        # Add start/end loss annotation for quick reference
        ax.annotate(
            f"Start: {losses[0]:.3f}",
            xy=(steps[0], losses[0]),
            xytext=(steps[0] + (steps[-1] - steps[0]) * 0.05, losses[0]),
            fontsize=10,
            color="#666666",
            arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8),
        )
        ax.annotate(
            f"End: {ema_losses[-1]:.3f}",
            xy=(steps[-1], ema_losses[-1]),
            xytext=(steps[-1] - (steps[-1] - steps[0]) * 0.2, ema_losses[-1]),
            fontsize=10,
            color=COLORS["loss_ema"],
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["loss_ema"], lw=0.8),
        )

        fig.tight_layout()

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"✅ Training loss plot saved to {output_path}")


def plot_metrics_comparison(
    eval_results: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Plot grouped bar chart comparing baseline vs. fine-tuned evaluation metrics.

    WHY two subplots instead of one:
      - ROUGE-L and BLEU are both on a 0–1 scale (or 0–100), so they can
        share an axis. Perplexity is on a completely different scale (often
        10–1000+). Plotting them on the same axis would either squash the
        small metrics or make perplexity invisible.
      - Using subplots (side by side) keeps each metric readable while
        maintaining a single cohesive figure.

    WHY value labels on bars:
      - Bar heights alone are hard to read precisely. Adding the numeric value
        on top of each bar lets viewers compare exact numbers without squinting
        at the y-axis. This is a standard practice in ML papers.

    Design choices:
      - Gray for baseline: muted, recedes visually → "this is what we had"
      - Teal/orange for fine-tuned: vibrant, draws attention → "this is the improvement"
      - Perplexity gets a different accent color (orange) because lower is
        better — the color distinction prevents viewers from misreading the chart.

    Args:
        eval_results: Dict with 'baseline' and 'finetuned' keys, each containing
                      metric names and values. Produced by evaluate.py.
        output_path: Where to save the PNG file.
    """
    # --- Extract metrics ---
    baseline = eval_results.get("baseline", {})
    finetuned = eval_results.get("finetuned", {})

    if not baseline or not finetuned:
        logger.error(
            "eval_results.json must contain 'baseline' and 'finetuned' keys.\n"
            "  → Make sure evaluate.py has been run and completed successfully."
        )
        return

    # Separate quality metrics (0–1 scale) from perplexity (unbounded scale)
    quality_metrics = ["rouge_l", "bleu"]
    quality_labels = ["ROUGE-L", "BLEU"]

    # Collect available quality metrics
    available_quality = []
    available_labels = []
    baseline_quality_vals = []
    finetuned_quality_vals = []

    for metric, label in zip(quality_metrics, quality_labels):
        if metric in baseline and metric in finetuned:
            available_quality.append(metric)
            available_labels.append(label)
            baseline_quality_vals.append(baseline[metric])
            finetuned_quality_vals.append(finetuned[metric])

    has_perplexity = "perplexity" in baseline and "perplexity" in finetuned

    if not available_quality and not has_perplexity:
        logger.error(
            "No recognized metrics found in eval_results.json.\n"
            "  → Expected keys: rouge_l, bleu, perplexity"
        )
        return

    # --- Determine subplot layout ---
    # WHY dynamic layout: if perplexity is missing (e.g., evaluation failed
    # partway), we still produce a useful chart with available metrics.
    if available_quality and has_perplexity:
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(12, 5),
            gridspec_kw={"width_ratios": [2, 1]},
        )
    elif available_quality:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = None
    else:
        fig, ax2 = plt.subplots(figsize=(5, 5))
        ax1 = None

    with plt.rc_context(STYLE_CONFIG):
        # --- Subplot 1: Quality Metrics (ROUGE-L, BLEU) ---
        if ax1 is not None and available_quality:
            x = np.arange(len(available_labels))
            bar_width = 0.35

            bars_base = ax1.bar(
                x - bar_width / 2,
                baseline_quality_vals,
                bar_width,
                label="Baseline",
                color=COLORS["baseline"],
                edgecolor="white",
                linewidth=0.8,
            )
            bars_ft = ax1.bar(
                x + bar_width / 2,
                finetuned_quality_vals,
                bar_width,
                label="Fine-Tuned",
                color=COLORS["finetuned"],
                edgecolor="white",
                linewidth=0.8,
            )

            # Add value labels on top of each bar
            _add_bar_labels(ax1, bars_base, fmt=".3f")
            _add_bar_labels(ax1, bars_ft, fmt=".3f", bold=True)

            ax1.set_xlabel("Metric")
            ax1.set_ylabel("Score")
            ax1.set_title("Quality Metrics (↑ higher is better)")
            ax1.set_xticks(x)
            ax1.set_xticklabels(available_labels)
            ax1.legend(loc="upper left")

            # Set y-axis to start at 0 for honest visual comparison
            ax1.set_ylim(bottom=0)
            # Add a bit of headroom for the value labels
            max_val = max(baseline_quality_vals + finetuned_quality_vals)
            ax1.set_ylim(top=max_val * 1.25)

        # --- Subplot 2: Perplexity ---
        if ax2 is not None and has_perplexity:
            ppl_base = baseline["perplexity"]
            ppl_ft = finetuned["perplexity"]

            x = np.arange(1)
            bar_width = 0.35

            bar_ppl_base = ax2.bar(
                x - bar_width / 2,
                [ppl_base],
                bar_width,
                label="Baseline",
                color=COLORS["perplexity_base"],
                edgecolor="white",
                linewidth=0.8,
            )
            bar_ppl_ft = ax2.bar(
                x + bar_width / 2,
                [ppl_ft],
                bar_width,
                label="Fine-Tuned",
                color=COLORS["perplexity_ft"],
                edgecolor="white",
                linewidth=0.8,
            )

            # Value labels
            _add_bar_labels(ax2, bar_ppl_base, fmt=".1f")
            _add_bar_labels(ax2, bar_ppl_ft, fmt=".1f", bold=True)

            ax2.set_xlabel("Metric")
            ax2.set_ylabel("Perplexity")
            ax2.set_title("Perplexity (↓ lower is better)")
            ax2.set_xticks(x)
            ax2.set_xticklabels(["Perplexity"])
            ax2.legend(loc="upper right")

            # Start y-axis at 0
            ax2.set_ylim(bottom=0)
            max_ppl = max(ppl_base, ppl_ft)
            ax2.set_ylim(top=max_ppl * 1.25)

        # --- Overall title ---
        fig.suptitle(
            "Baseline vs Fine-Tuned Model — Evaluation Metrics",
            fontsize=15,
            fontweight="bold",
            y=1.02,
        )

        fig.tight_layout()

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"✅ Metrics comparison plot saved to {output_path}")


def _add_bar_labels(
    ax: plt.Axes,
    bars: plt.bar,
    fmt: str = ".2f",
    bold: bool = False,
) -> None:
    """
    Add value labels on top of each bar in a bar chart.

    WHY this helper exists:
      - Matplotlib doesn't add value labels by default, and the boilerplate
        is verbose. Extracting it into a helper keeps the main plotting
        functions clean and focused on layout logic.
      - Bold labels on fine-tuned bars draw the eye to the improved values.

    Args:
        ax: The matplotlib Axes object containing the bars.
        bars: The bar container returned by ax.bar().
        fmt: Format string for the numeric label (e.g., ".3f" for 3 decimals).
        bold: Whether to render the label in bold (used for fine-tuned bars).
    """
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02,
            f"{height:{fmt}}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold" if bold else "normal",
            color="#333333",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Main entry point: load data, generate plots, save to disk.

    The script is designed to be fault-tolerant: if one input file is missing,
    it still attempts to generate the other plot. This way you can visualize
    training progress before evaluation is complete.
    """
    parser = argparse.ArgumentParser(
        description="Generate publication-quality plots for Medical Q&A fine-tuning results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML file (used for results directory path)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help=(
            "Directory containing training_log.json and eval_results.json. "
            "Overrides output_dir from config.yaml."
        ),
    )
    args = parser.parse_args()

    # --- Resolve results directory ---
    # WHY allow both config and CLI override:
    #   - Config provides the default for standard runs.
    #   - --results-dir lets you point at a different experiment without editing config.
    if args.results_dir is not None:
        results_dir = args.results_dir
    else:
        try:
            config = load_config(args.config)
            results_dir = config.get("output_dir", "./results")
        except FileNotFoundError:
            logger.warning(
                f"Config file not found: {args.config}. "
                f"Using default results directory: ./results"
            )
            results_dir = "./results"

    logger.info(f"Results directory: {results_dir}")

    # --- Track how many plots were generated ---
    plots_generated = 0

    # --- Plot 1: Training Loss Curve ---
    training_log_path = os.path.join(results_dir, "training_log.json")
    training_log = load_json_file(training_log_path)

    if training_log is not None:
        output_path = os.path.join(results_dir, "training_loss.png")
        plot_training_loss(training_log, output_path)
        plots_generated += 1
    else:
        logger.info(
            "⏭  Skipping training loss plot (training_log.json not found).\n"
            "   Run train.py first to generate training logs."
        )

    # --- Plot 2: Metrics Comparison ---
    eval_results_path = os.path.join(results_dir, "eval_results.json")
    eval_results = load_json_file(eval_results_path)

    if eval_results is not None:
        output_path = os.path.join(results_dir, "metrics_comparison.png")
        plot_metrics_comparison(eval_results, output_path)
        plots_generated += 1
    else:
        logger.info(
            "⏭  Skipping metrics comparison plot (eval_results.json not found).\n"
            "   Run evaluate.py first to generate evaluation results."
        )

    # --- Summary ---
    if plots_generated == 0:
        logger.warning(
            "\n⚠  No plots generated. Make sure you've run the pipeline:\n"
            "   1. python data_prep.py      → Prepare dataset\n"
            "   2. python train.py           → Train model (creates training_log.json)\n"
            "   3. python evaluate.py        → Evaluate model (creates eval_results.json)\n"
            "   4. python plot_results.py    → Generate plots (this script)"
        )
    elif plots_generated == 1:
        logger.info(
            f"\n📊 Generated {plots_generated}/2 plots. "
            f"Run the remaining pipeline step to generate the other."
        )
    else:
        logger.info(
            f"\n✅ All {plots_generated} plots generated successfully!\n"
            f"   📈 {os.path.join(results_dir, 'training_loss.png')}\n"
            f"   📊 {os.path.join(results_dir, 'metrics_comparison.png')}"
        )


if __name__ == "__main__":
    main()
