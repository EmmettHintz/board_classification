#!/usr/bin/env python
"""
evaluate_fen.py

Standalone evaluation script for the chess FEN prediction pipeline.

Usage:
    python evaluate_fen.py --boards-dir boards/ --fen-mapping fen_mapping.txt --model models/chess_piece_model.pkl --verbose

Computes exact-match, Levenshtein ratio, ROUGE-1, and square-level accuracy metrics.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

# Enable MPS fallback for MacOS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter

# Import the prediction function from your pipeline module
from chess_fen_fastai import predict_fen_from_image


# ----- Evaluation Utilities -----
def fen_to_grid(fen: str) -> list[str]:
    """
    Expand FEN "board part" into a flat list of 64 symbols.
    Digits become that many 'empty' entries; letters remain as-is.
    """
    # Extract just the board part if full FEN is provided
    board = fen.split()[0]
    grid = []

    # Count how many squares we're creating to help with debugging
    rank_count = 0

    for rank in board.split("/"):
        rank_squares = []
        for ch in rank:
            if ch.isdigit():
                rank_squares.extend(["empty"] * int(ch))
            else:
                rank_squares.append(ch)

        rank_count += 1
        # Check if this rank has exactly 8 squares
        if len(rank_squares) != 8:
            raise ValueError(
                f"Rank {rank_count} ('{rank}') expands to {len(rank_squares)} squares, expected 8"
            )

        grid.extend(rank_squares)

    # Check if we have exactly 8 ranks
    if rank_count != 8:
        raise ValueError(f"FEN has {rank_count} ranks, expected 8: '{fen}'")

    # Final check for total squares
    if len(grid) != 64:
        raise ValueError(f"Expanded FEN has {len(grid)} squares, expected 64: '{fen}'")

    return grid


def rouge1(pred: str, ref: str) -> tuple[float, float, float]:
    """
    Unigram-overlap ROUGE-1 precision, recall, and F1 on character tokens.
    """
    pgrams = list(pred)
    rgrams = list(ref)
    pc = Counter(pgrams)
    rc = Counter(rgrams)
    overlap = sum(min(pc[g], rc[g]) for g in pc)
    prec = overlap / len(pgrams) if pgrams else 0.0
    rec = overlap / len(rgrams) if rgrams else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def evaluate_batch(preds: list[str], truths: list[str]) -> dict[str, float]:
    """
    Given lists of predicted and true FEN strings, compute:
      - exact_match: fraction exactly equal
      - levenshtein_avg: average SequenceMatcher ratio
      - rouge1_prec, rouge1_rec, rouge1_f1
      - square_acc: average per-board square-level accuracy
    """
    n = len(preds)
    if n == 0:
        raise ValueError("No boards to evaluate.")

    exact = 0
    lev_sum = 0.0
    r_prec = r_rec = r_f1 = 0.0
    sq_acc_sum = 0.0

    for p, t in zip(preds, truths):
        # Exact match
        if p == t:
            exact += 1
        # Levenshtein ratio
        lev_sum += SequenceMatcher(None, p, t).ratio()
        # ROUGE-1
        pr, rc, pf1 = rouge1(p, t)
        r_prec += pr
        r_rec += rc
        r_f1 += pf1
        # Square-level accuracy
        pg = fen_to_grid(p)
        tg = fen_to_grid(t)
        matches = sum(1 for a, b in zip(pg, tg) if a == b)
        sq_acc_sum += matches / 64.0

    return {
        "exact_match": exact / n,
        "levenshtein_avg": lev_sum / n,
        "rouge1_prec": r_prec / n,
        "rouge1_rec": r_rec / n,
        "rouge1_f1": r_f1 / n,
        "square_acc": sq_acc_sum / n,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FEN predictions against ground-truth mapping"
    )
    parser.add_argument(
        "--boards-dir",
        required=True,
        help="Directory containing board images (PDF/PNG)",
    )
    parser.add_argument(
        "--fen-mapping",
        required=True,
        help="CSV or text file with lines 'board_name,fen_string'",
    )
    parser.add_argument(
        "--model",
        default="models/chess_piece_model.pkl",
        help="Path to the trained model for prediction",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-board predictions and ground truth",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip boards with invalid FEN strings instead of failing",
    )
    args = parser.parse_args()

    # Load ground-truth FEN mapping
    fen_map: dict[str, str] = {}
    with open(args.fen_mapping, "r") as f:
        for line in f:
            if not line.strip():
                continue
            name, fen = line.strip().split(",", 1)
            fen_map[name.strip()] = fen.strip()

    # Validate all ground truth FEN strings first
    if not args.skip_invalid:
        for name, fen in fen_map.items():
            try:
                _ = fen_to_grid(fen)
            except ValueError as e:
                print(f"Error in ground truth FEN for {name}: {e}")
                print("Use --skip-invalid to continue anyway, or fix the FEN mapping.")
                return

    preds = []
    truths = []
    failed = 0
    skipped = 0

    # Iterate boards
    boards_path = Path(args.boards_dir)
    for img_path in sorted(boards_path.glob("*.pdf")) + sorted(
        boards_path.glob("*.png")
    ):
        board_name = img_path.stem
        if board_name not in fen_map:
            if args.verbose:
                print(f"Skipping '{board_name}': no ground truth FEN")
            continue

        true_fen = fen_map[board_name]

        # Validate the true FEN again
        try:
            _ = fen_to_grid(true_fen)
        except ValueError as e:
            print(f"Invalid ground truth FEN for {board_name}: {e}")
            skipped += 1
            continue

        # Get prediction
        pred_fen = predict_fen_from_image(str(img_path), args.model)

        # Handle failed prediction
        if pred_fen is None:
            failed += 1
            if args.verbose:
                print(
                    f"{board_name} | Prediction FAILED\n            True: {true_fen}\n"
                )
            continue

        # Validate prediction FEN format
        try:
            _ = fen_to_grid(pred_fen)
        except ValueError as e:
            failed += 1
            if args.verbose:
                print(
                    f"{board_name} | Malformed prediction: {pred_fen}\n            True: {true_fen}\n            Error: {e}"
                )
            continue

        preds.append(pred_fen)
        truths.append(true_fen)

        if args.verbose:
            print(f"{board_name} | Pred: {pred_fen}\n            True: {true_fen}\n")

    # Compute metrics
    if preds:
        try:
            metrics = evaluate_batch(preds, truths)
            print("\n=== Evaluation Results ===")
            for k, v in metrics.items():
                print(f"{k:15}: {v:.4f}")

            # Create output directory if it doesn't exist
            output_dir = Path("evaluation_output")
            output_dir.mkdir(exist_ok=True)

            # Add visualization
            plt.figure(figsize=(6, 6))
            metrics_to_plot = {
                k: v
                for k, v in metrics.items()
                if k != "levenshtein_avg" and k != "rouge1_prec" and k != "rouge1_rec"
            }

            # Bar chart of metrics with flare color palette
            ax = sns.barplot(
                x=list(metrics_to_plot.keys()),
                y=list(metrics_to_plot.values()),
                palette="flare",
            )
            plt.ylim(0, 1.0)

            plt.title("Model Performance Metrics")
            plt.ylabel("Score (higher is better)")

            # Put bar labels on interior of bars
            for i, v in enumerate(metrics_to_plot.values()):
                ax.text(
                    i, v / 2, f"{v:.3f}", ha="center", color="white", fontweight="bold"
                )

            plt.tight_layout()
            output_path = output_dir / "evaluation_metrics.png"

            plt.savefig(output_path, dpi=300)
            print(f"Visualization saved as '{output_path}'")
        except Exception as e:
            print(f"Error computing metrics: {e}")
    else:
        print("No successful predictions to evaluate.")

    if failed > 0 or skipped > 0:
        print(
            f"\n{failed} boards failed prediction, {skipped} boards skipped due to invalid ground truth."
        )


if __name__ == "__main__":
    main()
