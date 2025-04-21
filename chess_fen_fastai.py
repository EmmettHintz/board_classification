import os

# Set PyTorch MPS fallback for MacOS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import numpy as np
from pathlib import Path
import shutil
import argparse
from pdf2image import convert_from_path
from fastai.vision.all import *
from fastai.metrics import error_rate
import torch

# ----- Configuration -----
BOARD_SIZE = 800
# Chess piece labels: "empty" for empty squares, "_white" suffix for white pieces, "_black" suffix for black pieces
# This mapping will convert from FEN notation (P, N, B, etc.) to directory names (P_white, N_white, etc.)
CLASS_LABELS = [
    "empty",
    "P_white",
    "N_white",
    "B_white",
    "R_white",
    "Q_white",
    "K_white",
    "p_black",
    "n_black",
    "b_black",
    "r_black",
    "q_black",
    "k_black",
]

# Mapping from FEN notation to directory names
FEN_TO_LABEL = {
    "P": "P_white",
    "N": "N_white",
    "B": "B_white",
    "R": "R_white",
    "Q": "Q_white",
    "K": "K_white",
    "p": "p_black",
    "n": "n_black",
    "b": "b_black",
    "r": "r_black",
    "q": "q_black",
    "k": "k_black",
}


# ----- PDF Conversion -----
def pdf_to_image(path: str, dpi: int = 200) -> np.ndarray:
    """Convert first page of a PDF to BGR image."""
    pages = convert_from_path(path, dpi=dpi)
    arr = np.array(pages[0])
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ----- Preprocessing Utilities -----
def preprocess_board_image(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Enhance and binarize the input image for robust contour detection.
    Returns binary mask (closed grid) and CLAHE-enhanced gray.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    bin_adapt = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    _, bin_otsu = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    binary = cv2.bitwise_or(bin_adapt, bin_otsu)
    k_small = np.ones((2, 2), np.uint8)
    k_large = np.ones((3, 3), np.uint8)
    denoised = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_small)
    dilated = cv2.dilate(denoised, k_large, iterations=1)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, k_large, iterations=2)
    return closed, enhanced


# ----- Contour & Corner Detection -----
def find_board_contour(bin_img: np.ndarray) -> np.ndarray:
    """Find the best 4-point contour approximating the board."""
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("No board contour found")
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    best, best_score = cnts[0], float("inf")
    for c in cnts[:10]:  # Try more contours (increased from 5)
        area = cv2.contourArea(c)
        if area < 0.1 * cv2.contourArea(cnts[0]):  # More permissive threshold
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            rect_area = cv2.contourArea(np.int32(box))
            score = abs(1 - area / rect_area) + abs(16 - (peri * peri / area)) * 0.1
            if score < best_score:
                best_score, best = score, c
    peri = cv2.arcLength(best, True)
    approx = cv2.approxPolyDP(best, 0.02 * peri, True)
    if len(approx) != 4:
        # Force a rectangular approximation
        rect = cv2.minAreaRect(best)
        approx = np.int32(cv2.boxPoints(rect))
    return np.int32(approx)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points: tl, tr, br, bl."""
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


# ----- Segmentation Pipeline -----
def warp_and_segment(img: np.ndarray) -> list[list[np.ndarray]]:
    """Full pipeline: preprocess → contour → corners → subdivide → refine."""
    try:
        binary, enhanced = preprocess_board_image(img)
        contour = find_board_contour(binary)
        quad = order_points(contour)

        # Debug visualization - uncomment if needed
        # vis_img = img.copy()
        # cv2.drawContours(vis_img, [contour], -1, (0, 255, 0), 3)
        # cv2.imwrite("debug_contour.png", vis_img)

        tl, tr, br, bl = quad
        cell = BOARD_SIZE // 8
        grid = []
        for i in range(8):
            row = []
            v0, v1 = i / 8.0, (i + 1) / 8.0
            for j in range(8):
                u0, u1 = j / 8.0, (j + 1) / 8.0
                src = np.vstack(
                    [
                        (1 - u0) * (1 - v0) * tl
                        + u0 * (1 - v0) * tr
                        + u0 * v0 * br
                        + (1 - u0) * v0 * bl,
                        (1 - u1) * (1 - v0) * tl
                        + u1 * (1 - v0) * tr
                        + u1 * v0 * br
                        + (1 - u1) * v0 * bl,
                        (1 - u1) * (1 - v1) * tl
                        + u1 * (1 - v1) * tr
                        + u1 * v1 * br
                        + (1 - u1) * v1 * bl,
                        (1 - u0) * (1 - v1) * tl
                        + u0 * (1 - v1) * tr
                        + u0 * v1 * br
                        + (1 - u0) * v1 * bl,
                    ]
                ).astype("float32")
                dst = np.array(
                    [[0, 0], [cell, 0], [cell, cell], [0, cell]], dtype="float32"
                )
                M = cv2.getPerspectiveTransform(src, dst)
                patch = cv2.warpPerspective(img, M, (cell, cell))
                patch = refine_cell(patch)
                row.append(patch)
            grid.append(row)
        return grid
    except Exception as e:
        print(f"Error in board segmentation: {e}")
        # Return empty grid as fallback
        return []


# ----- Cell-Level Edge Refinement -----
def refine_cell(cell: np.ndarray, pad: int = 6) -> np.ndarray:
    """Crop each cell to its strongest local box via Hough-lines."""
    h, w = cell.shape[:2]
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=30, minLineLength=w // 3, maxLineGap=5
    )
    xs, ys = [], []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            if abs(x1 - x2) < 10:
                xs += [x1, x2]
            if abs(y1 - y2) < 10:
                ys += [y1, y2]
    xs += [0, w]
    ys += [0, h]
    x0, x1 = max(min(xs) - pad, 0), min(max(xs) + pad, w)
    y0, y1 = max(min(ys) - pad, 0), min(max(ys) + pad, h)
    return cell[y0:y1, x0:x1] if (x1 > x0 and y1 > y0) else cell


# ----- FEN Parsing Functions -----
def parse_fen_to_board(fen: str) -> list[list[str]]:
    """
    Parse a FEN string to a 2D board representation.
    Returns an 8x8 grid with piece symbols converted to our label format or 'empty'.
    """
    # Debug the FEN string
    print(f"Parsing FEN: {fen}")

    ranks = fen.split(" ")[0].split("/")
    board = []

    # Count occurrences for debugging
    piece_counts = {label: 0 for label in CLASS_LABELS}

    for rank in ranks:
        row = []
        for char in rank:
            if char.isdigit():
                # Add empty squares
                row.extend(["empty"] * int(char))
                piece_counts["empty"] += int(char)
            else:
                # Convert FEN notation to our label format
                if char in FEN_TO_LABEL:
                    label = FEN_TO_LABEL[char]
                    row.append(label)
                    piece_counts[label] += 1
                else:
                    print(f"Warning: Unknown piece symbol '{char}' in FEN")
                    row.append("empty")  # Default to empty if unknown
        board.append(row)

    # Print piece counts for debugging
    print("Piece counts from FEN:")
    for piece, count in piece_counts.items():
        if count > 0:
            print(f"  {piece}: {count}")

    return board


def create_output_structure(output_dir: str):
    """Create output directory structure for all chess piece types."""
    base_dir = Path(output_dir)
    if base_dir.exists():
        shutil.rmtree(base_dir)

    print("Creating directories for the following labels:")
    for label in CLASS_LABELS:
        print(f"  Creating directory for: {label}")
        label_dir = base_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        # Create an empty placeholder file to ensure directory is recognized
        if label != "empty" and label.islower():  # Only for black pieces
            placeholder_path = label_dir / ".placeholder"
            with open(placeholder_path, "w") as f:
                f.write(f"Placeholder for {label} pieces")

    # Verify all directories were created
    for label in CLASS_LABELS:
        dir_path = base_dir / label
        if dir_path.exists():
            print(f"  Verified: Directory {label}/ exists")
        else:
            print(f"  ERROR: Directory {label}/ was NOT created properly")

    return base_dir


# ----- Auto-Labeling Commands -----
def cmd_segment_and_label(boards_dir: str, out_dir: str, fen_mapping_file: str):
    """
    Segment chess boards and automatically label them based on FEN strings.

    Args:
        boards_dir: Directory with board PDFs
        out_dir: Output directory
        fen_mapping_file: File with board_name,fen_string mappings
    """
    # Load FEN mappings
    fen_mappings = {}
    with open(fen_mapping_file, "r") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(",", 1)
                if len(parts) == 2:
                    board_name, fen = parts
                    fen_mappings[board_name.strip()] = fen.strip()

    if not fen_mappings:
        print(f"No valid FEN mappings found in {fen_mapping_file}")
        return

    print(f"Loaded {len(fen_mappings)} FEN mappings")

    # Analyze FEN symbols for debugging
    print("\nChecking FEN strings for valid piece symbols...")
    all_symbols = set()
    symbol_counts = {"white": {}, "black": {}}

    for board_name, fen in fen_mappings.items():
        fen_board = fen.split(" ")[0]  # Get just the board part
        for char in fen_board:
            if char.isalpha():
                all_symbols.add(char)
                if char.isupper():  # White piece
                    symbol_counts["white"][char] = (
                        symbol_counts["white"].get(char, 0) + 1
                    )
                elif char.islower():  # Black piece
                    symbol_counts["black"][char] = (
                        symbol_counts["black"].get(char, 0) + 1
                    )

    print("All unique piece symbols found in FEN strings:")
    print(f"  {sorted(all_symbols)}")

    print("White piece counts across all boards:")
    for symbol, count in symbol_counts["white"].items():
        label = FEN_TO_LABEL.get(symbol, "unknown")
        print(f"  {symbol} ({label}): {count}")

    print("Black piece counts across all boards:")
    for symbol, count in symbol_counts["black"].items():
        label = FEN_TO_LABEL.get(symbol, "unknown")
        print(f"  {symbol} ({label}): {count}")

    # Create output structure
    output_dir = create_output_structure(out_dir)

    # Process each board
    src = Path(boards_dir)
    processed_count = 0
    error_count = 0

    for pdf in src.glob("*.pdf"):
        board_name = pdf.stem

        # Skip boards without FEN mapping
        if board_name not in fen_mappings:
            print(f"Skipping {board_name} (no FEN mapping)")
            continue

        try:
            # Parse FEN to get board state
            fen = fen_mappings[board_name]
            board_state = parse_fen_to_board(fen)

            # Segment board into cells
            print(f"Processing {board_name}...")
            img = pdf_to_image(str(pdf))
            print(f"  PDF image size: {img.shape}")

            cells = warp_and_segment(img)
            if not cells:
                print(f"  Error: Failed to segment {board_name}")
                error_count += 1
                continue

            print(
                f"  Segmented into {len(cells)} rows, {len(cells[0]) if cells else 0} cols"
            )

            # Save each cell with its label
            for i, (row_cells, row_labels) in enumerate(zip(cells, board_state)):
                for j, (cell, label) in enumerate(zip(row_cells, row_labels)):
                    # Add warning for label mismatch
                    if label not in CLASS_LABELS:
                        print(
                            f"Warning: label '{label}' not in CLASS_LABELS for {board_name} ({i},{j})"
                        )
                        continue
                    # Debug: print cell shape and type
                    print(
                        f"  Saving {board_name} ({i},{j}) label={label} cell shape={cell.shape} dtype={cell.dtype}"
                    )
                    label_dir = output_dir / label
                    cell_path = label_dir / f"{board_name}_{i}_{j}.png"
                    success = cv2.imwrite(str(cell_path), cell)
                    if not success:
                        print(f"  Failed to save image: {cell_path}")

            processed_count += 1
            print(f"Processed {board_name} with FEN: {fen}")

        except Exception as e:
            print(f"Error processing {board_name}: {e}")
            error_count += 1

    # Count how many images were saved in each directory
    print("\nFinal verification of output directories:")
    for label in CLASS_LABELS:
        dir_path = output_dir / label
        if not dir_path.exists():
            print(f"ERROR: Directory for {label} does not exist!")
            continue

        count = len(list(dir_path.glob("*.png")))
        print(f"{label}: {count} images")

        # For directories with no images, explain why
        if count == 0 and label.islower():
            print(
                f"  Note: No images for {label} (black piece) - Check if your FEN strings include this piece type"
            )

    print(
        f"\nProcessed {processed_count} boards successfully, {error_count} errors. Images organized by piece type."
    )
    if processed_count > 0:
        print(
            f"You can now train using: python chess_fen_fastai.py train --data-path {out_dir} --epochs 5"
        )
        print(f"For more options: python chess_fen_fastai.py train --help")
    else:
        print("No boards were processed successfully. Please check the errors above.")


# ----- Training Functions -----
def train_chess_classifier(
    data_path: str,
    epochs: int = 5,
    batch_size: int = 16,
    img_size: int = 224,
    learning_rate: float = 1e-3,
    arch: str = "resnet34",
    save_model: bool = True,
    output_model: str = "chess_piece_model.pkl",
    device: str = "auto",
):
    """
    Train a chess piece classifier using fastai.

    Args:
        data_path: Path to the directory with labeled images
        epochs: Number of epochs to train
        batch_size: Batch size for training
        img_size: Size to resize images to
        learning_rate: Learning rate
        arch: Model architecture to use ('resnet34', 'resnet50', etc.)
        save_model: Whether to save the model
        output_model: Filename to save the model to
        device: Device to use for training ('auto', 'cpu', 'cuda', 'mps')
    """
    print(f"Starting training with {epochs} epochs using {arch}...")

    # Configure device
    if device != "auto":
        if device in ["cpu", "cuda", "mps"]:
            print(f"Using device: {device}")
            torch.device(device)
            if device == "cpu":
                print("Forcing CPU usage regardless of GPU availability")
                # For fastai, setting these environment variables ensures CPU usage
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                os.environ["MPS_AVAILABLE"] = "0"
        else:
            print(f"Unknown device '{device}', falling back to auto-detection")

    # Setup path and verify it exists
    path = Path(data_path)
    if not path.exists():
        raise ValueError(f"Data path {data_path} does not exist!")

    # Check if directories are present
    dirs = [d for d in path.iterdir() if d.is_dir()]
    if not dirs:
        raise ValueError(f"No class directories found in {data_path}!")

    print(f"Found {len(dirs)} class directories: {[d.name for d in dirs]}")

    # Setup DataBlock
    chess_data = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(img_size),
        batch_tfms=[
            *aug_transforms(size=img_size),
            Normalize.from_stats(*imagenet_stats),
        ],
    )

    # Create DataLoaders
    try:
        dls = chess_data.dataloaders(path, bs=batch_size)
        print(f"Training set: {len(dls.train_ds)} images")
        print(f"Validation set: {len(dls.valid_ds)} images")

        # Show batch if running in a notebook
        # dls.show_batch(max_n=9, figsize=(8, 8))
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        return

    # Create learner
    learn = vision_learner(dls, getattr(models, arch), metrics=error_rate)

    # Find good learning rate
    # learn.lr_find()

    # Train with one cycle policy
    learn.fine_tune(epochs, learning_rate)

    # Print results
    print("\nTraining complete!")
    valid_loss, accuracy = learn.validate()
    print(f"Validation loss: {valid_loss:.4f}")
    print(f"Accuracy: {(1-accuracy):.4f} ({(1-accuracy)*100:.1f}%)")

    interp = ClassificationInterpretation.from_learner(learn)

    # 1. print raw confusion matrix array
    cm = interp.confusion_matrix()
    print("\nConfusion Matrix:")
    print(cm)

    # 2. plot it
    interp.plot_confusion_matrix(figsize=(6, 6), dpi=80)

    # Confusion matrix (just compute, don't display)
    cm = interp.confusion_matrix()
    print(f"Confusion matrix shape: {cm.shape}")

    # Save the model
    if save_model:
        # Save the trained model
        model_path = Path("./models")
        model_path.mkdir(exist_ok=True)

        full_path = model_path / output_model
        print(f"Saving model to {full_path}")
        learn.export(full_path)

    return learn


# ----- Prediction Functions -----
def predict_fen_from_image(
    image_path: str, model_path: str = "models/chess_piece_model.pkl"
):
    """
    Predict FEN notation from a chess board image.

    Args:
        image_path: Path to the chess board image (PDF or PNG)
        model_path: Path to the trained model

    Returns:
        FEN notation string
    """
    # Load the model
    print(f"Loading model from {model_path}...")
    try:
        learn = load_learner(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Load and prepare the image
    print(f"Processing image: {image_path}")
    if image_path.lower().endswith(".pdf"):
        img = pdf_to_image(image_path)
    else:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return None

    # Segment the board
    cells = warp_and_segment(img)
    if not cells or len(cells) != 8 or any(len(row) != 8 for row in cells):
        print(f"Error: Failed to properly segment board into 8x8 grid")
        return None

    print("Board segmented, predicting pieces...")

    # Create a matrix to store predictions
    predictions = []

    # Process each cell
    for i, row in enumerate(cells):
        pred_row = []
        for j, cell in enumerate(row):
            # Convert OpenCV BGR to RGB for fastai
            cell_rgb = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)

            # Predict
            pred, _, probs = learn.predict(cell_rgb)
            pred_row.append(str(pred))

            # Print confidence for debugging
            confidence = probs.max().item()
            print(f"Square {chr(97+j)}{8-i}: {pred} (confidence: {confidence:.2f})")

        predictions.append(pred_row)

    # Convert predictions to FEN notation
    fen = generate_fen_from_predictions(predictions)
    print(f"FEN: {fen}")

    return fen


def generate_fen_from_predictions(predictions):
    """
    Convert a 2D grid of piece predictions to FEN notation.

    Args:
        predictions: 8x8 grid of piece labels (e.g., 'P_white', 'empty', etc.)

    Returns:
        FEN notation string
    """
    # Reverse mapping from our label format to FEN notation
    LABEL_TO_FEN = {v: k for k, v in FEN_TO_LABEL.items()}
    LABEL_TO_FEN["empty"] = ""

    fen_parts = []

    for row in predictions:
        fen_row = ""
        empty_count = 0

        for cell in row:
            if cell == "empty":
                empty_count += 1
            else:
                # If we had empty squares before this piece, add the count
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0

                # Add the piece symbol
                fen_row += LABEL_TO_FEN.get(cell, "")

        # If the row ends with empty squares
        if empty_count > 0:
            fen_row += str(empty_count)

        fen_parts.append(fen_row)

    # Join rows with slashes
    board_fen = "/".join(fen_parts)

    # For now, just include the board position, assume it's white's turn with full castling rights
    return f"{board_fen}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chess board auto-labeler and classifier using FEN notation"
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Command to run", required=True
    )

    # Segmentation and labeling command
    segment_parser = subparsers.add_parser(
        "segment", help="Segment and label chess boards"
    )
    segment_parser.add_argument(
        "--boards-dir", required=True, help="Directory with board PDFs"
    )
    segment_parser.add_argument("--out-dir", required=True, help="Output directory")
    segment_parser.add_argument(
        "--fen-mapping", required=True, help="File with board_name,fen_string mappings"
    )

    # Training command
    train_parser = subparsers.add_parser("train", help="Train a chess piece classifier")
    train_parser.add_argument(
        "--data-path", required=True, help="Path to labeled data directory"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs to train"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training"
    )
    train_parser.add_argument(
        "--img-size", type=int, default=224, help="Image size for training"
    )
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--arch", default="resnet34", help="Model architecture")
    train_parser.add_argument(
        "--output-model", default="chess_piece_model.pkl", help="Output model filename"
    )
    train_parser.add_argument(
        "--device",
        default="auto",
        help="Device to use for training ('auto', 'cpu', 'cuda', 'mps')",
    )

    # Prediction command
    predict_parser = subparsers.add_parser(
        "predict", help="Predict FEN notation from a chess board image"
    )
    predict_parser.add_argument(
        "--image", required=True, help="Path to the chess board image (PDF or PNG)"
    )
    predict_parser.add_argument(
        "--model",
        default="models/chess_piece_model.pkl",
        help="Path to the trained model",
    )

    args = parser.parse_args()

    # Execute the appropriate command
    if args.command == "segment":
        cmd_segment_and_label(args.boards_dir, args.out_dir, args.fen_mapping)
    elif args.command == "train":
        train_chess_classifier(
            data_path=args.data_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            learning_rate=args.lr,
            arch=args.arch,
            output_model=args.output_model,
            device=args.device,
        )
    elif args.command == "predict":
        predict_fen_from_image(args.image, args.model)
