import os
import cv2
import numpy as np
from pathlib import Path
import shutil
import argparse
from pdf2image import convert_from_path

# ----- Configuration -----
BOARD_SIZE = 800
CLASS_LABELS = ["empty", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]


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
    for c in cnts[:5]:
        area = cv2.contourArea(c)
        if area < 0.2 * cv2.contourArea(cnts[0]):
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
        rect = cv2.minAreaRect(best)
        approx = cv2.boxPoints(rect)
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
    binary, enhanced = preprocess_board_image(img)
    contour = find_board_contour(binary)
    quad = order_points(contour)
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
    Returns an 8x8 grid with piece symbols or 'empty'.
    """
    ranks = fen.split(" ")[0].split("/")
    board = []

    for rank in ranks:
        row = []
        for char in rank:
            if char.isdigit():
                # Add empty squares
                row.extend(["empty"] * int(char))
            else:
                # Add piece
                row.append(char)
        board.append(row)

    return board


def create_output_structure(output_dir: str):
    """Create output directory structure for all chess piece types."""
    base_dir = Path(output_dir)
    if base_dir.exists():
        shutil.rmtree(base_dir)

    for label in CLASS_LABELS:
        (base_dir / label).mkdir(parents=True, exist_ok=True)

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
                    fen_mappings[board_name] = fen

    if not fen_mappings:
        print(f"No valid FEN mappings found in {fen_mapping_file}")
        return

    print(f"Loaded {len(fen_mappings)} FEN mappings")

    # Create output structure
    output_dir = create_output_structure(out_dir)

    # Process each board
    src = Path(boards_dir)
    processed_count = 0

    for pdf in src.glob("*.pdf"):
        board_name = pdf.stem

        # Skip boards without FEN mapping
        if board_name not in fen_mappings:
            print(f"Skipping {board_name} (no FEN mapping)")
            continue

        # Parse FEN to get board state
        fen = fen_mappings[board_name]
        board_state = parse_fen_to_board(fen)

        # Segment board into cells
        img = pdf_to_image(str(pdf))
        cells = warp_and_segment(img)

        # Save each cell with its label
        for i, (row_cells, row_labels) in enumerate(zip(cells, board_state)):
            for j, (cell, label) in enumerate(zip(row_cells, row_labels)):
                # Save cell to appropriate class directory
                label_dir = output_dir / label
                cell_path = label_dir / f"{board_name}_{i}_{j}.png"
                cv2.imwrite(str(cell_path), cell)

        processed_count += 1
        print(f"Processed {board_name} with FEN: {fen}")

    # Count how many images were saved in each directory
    for label in CLASS_LABELS:
        dir_path = output_dir / label
        count = len(list(dir_path.glob("*.png")))
        print(f"{label}: {count} images")

    print(f"\nProcessed {processed_count} boards. Images organized by piece type.")
    print(
        f"You can now train using: python chess_fen_fastai.py train --data-path {out_dir} --epochs 5"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chess board auto-labeler using FEN notation"
    )
    parser.add_argument("--boards-dir", required=True, help="Directory with board PDFs")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument(
        "--fen-mapping", required=True, help="File with board_name,fen_string mappings"
    )

    args = parser.parse_args()
    cmd_segment_and_label(args.boards_dir, args.out_dir, args.fen_mapping)
