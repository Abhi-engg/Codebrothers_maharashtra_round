#!/usr/bin/env python3
"""
Re:Learn — Unified Training CLI Entrypoint
Author: Senior ML & DL Training Engineer

Provides a unified command-line interface to train and evaluate:
1. Fast Baseline Model (TF-IDF + Calibrated Logistic Regression)
2. DeBERTa-v3 with Partial Training Strategy (@[Quote])

Examples:
  # Train fast baseline model
  python train.py --model baseline

  # Train DeBERTa-v3 model with Partial Training Strategy
  python train.py --model deberta --epochs 3 --batch-size 16 --loss focal

  # Quick smoke test on baseline
  python train.py --model baseline --max-train-samples 100 --max-val-samples 50

  # Quick smoke test on DeBERTa
  python train.py --model deberta --max-train-samples 100 --max-val-samples 50 --epochs 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    # Ensure UTF-8 output on Windows console
    if sys.platform == "win32":
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")

    # If --model is specified along with -h/--help, forward directly to sub-script help
    argv_copy = list(sys.argv)
    model_choice = None
    if "--model" in argv_copy:
        idx = argv_copy.index("--model")
        if idx + 1 < len(argv_copy):
            model_choice = argv_copy[idx + 1]

    if model_choice in ("baseline", "deberta") and any(h in argv_copy for h in ("-h", "--help")):
        # Remove --model and its value so sub-script gets the rest (including --help)
        remaining = [arg for i, arg in enumerate(argv_copy) if i not in (idx, idx + 1)]
        sys.argv = [sys.argv[0]] + remaining[1:]
        if model_choice == "baseline":
            import train_baseline
            return train_baseline.main()
        else:
            import train_deberta
            return train_deberta.main()

    parser = argparse.ArgumentParser(
        description="Re:Learn Unified Model Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models available:
  baseline : Scikit-learn TF-IDF + Calibrated Logistic Regression (Fast CPU/GPU baseline)
  deberta  : Microsoft DeBERTa-v3-base with Partial Training Strategy (@[Quote] freezing & Focal Loss)

Run with '--help' after specifying model (e.g. 'python train.py --model baseline --help') or inspect train_baseline.py / train_deberta.py.
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "deberta"],
        required=True,
        help="Model architecture to train ('baseline' or 'deberta')",
    )

    # Parse only known args for --model so that remaining args can be forwarded to the specific runner
    known_args, remaining_args = parser.parse_known_args()

    if known_args.model == "baseline":
        import train_baseline
        # Re-set sys.argv for the target script
        sys.argv = [sys.argv[0]] + remaining_args
        return train_baseline.main()

    elif known_args.model == "deberta":
        import train_deberta
        # Re-set sys.argv for the target script
        sys.argv = [sys.argv[0]] + remaining_args
        return train_deberta.main()

    else:
        print(f"Error: Unsupported model '{known_args.model}'", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
