# chess_fen_fastai.py
"""
End-to-end hand-drawn chessboard → FEN toolkit using fastai.

Commands:
  • segment:   PDF boards → flat per-cell images + folder skeleton
  • prepare:   PDF boards + JSON FEN labels → per-class training images
  • train:     train classifier on per-class images
  • infer:     classify board PDF/image → FEN
"""
import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from fastai.vision.all import *
from fastai.data.all import get_image_files, parent_label, RandomSplitter
from fastai.callback.all import SaveModelCallback, EarlyStoppingCallback, MixUp
from pdf2image import convert_from_path

# ----- Configuration -----
BOARD_SIZE = 800
CELL_SIZE = BOARD_SIZE // 8
CLASS_LABELS = ["empty", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]

# ----- Segmentation Utilities -----


def pdf_to_image(path: str, dpi: int = 200) -> np.ndarray:
    pages = convert_from_path(path, dpi=dpi)
    return cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)


def find_board(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype("float32")
    s, diff = box.sum(axis=1), np.diff(box, axis=1)
    tl, br = box[np.argmin(s)], box[np.argmax(s)]
    tr, bl = box[np.argmin(diff)], box[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


def warp_and_segment(img: np.ndarray) -> list[list[np.ndarray]]:
    pts = find_board(img)
    dst = np.array(
        [[0, 0], [BOARD_SIZE, 0], [BOARD_SIZE, BOARD_SIZE], [0, BOARD_SIZE]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(pts, dst)
    warp = cv2.warpPerspective(img, M, (BOARD_SIZE, BOARD_SIZE))
    grid = []
    for i in range(8):
        row = []
        for j in range(8):
            cell = warp[
                i * CELL_SIZE : (i + 1) * CELL_SIZE, j * CELL_SIZE : (j + 1) * CELL_SIZE
            ]
            row.append(cell)
        grid.append(row)
    return grid


# ----- FEN Utilities -----


def fen_to_matrix(fen: str) -> list[list[str]]:
    rows = fen.split()[0].split("/")
    mat = []
    for r in rows:
        row = []
        for c in r:
            if c.isdigit():
                row += ["empty"] * int(c)
            else:
                row.append(c)
        mat.append(row)
    return mat


# ----- CLI Commands -----


def cmd_segment(boards_dir: str, out_dir: str):
    """Segment all PDFs into flat cell images and create class-folder skeleton."""
    src = Path(boards_dir)
    dst = Path(out_dir)
    allcells = dst / "all_cells"
    allcells.mkdir(parents=True, exist_ok=True)
    for pdf in src.glob("*.pdf"):
        img = pdf_to_image(str(pdf))
        cells = warp_and_segment(img)
        for i, row in enumerate(cells):
            for j, cell in enumerate(row):
                fname = f"{pdf.stem}_{i}_{j}.png"
                cv2.imwrite(str(allcells / fname), cell)
        print(f"Segmented {pdf.name}")
    # create empty class dirs
    for split in ["train", "valid"]:
        for cls in CLASS_LABELS:
            (dst / split / cls).mkdir(parents=True, exist_ok=True)
    print(f"Flattened cells -> {allcells} and skeleton under {dst}/train, {dst}/valid.")


def cmd_prepare(boards_dir: str, labels_json: str, out_dir: str):
    """Use FEN labels JSON to generate per-class training images."""
    src = Path(boards_dir)
    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    with open(labels_json) as f:
        label_map = json.load(f)
    for pdf_name, fen in label_map.items():
        pdf = src / pdf_name
        if not pdf.exists():
            continue
        mat = fen_to_matrix(fen)
        cells = warp_and_segment(pdf_to_image(str(pdf)))
        for i in range(8):
            for j in range(8):
                lbl = mat[i][j]
                dest = dst / lbl
                dest.mkdir(parents=True, exist_ok=True)
                fname = f"{pdf.stem}_{i}_{j}.png"
                cv2.imwrite(str(dest / fname), cells[i][j])
        print(f"Prepared {pdf_name}")


def get_dls(path: Path, img_size: int = 224, bs: int = 64):
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(img_size),
        batch_tfms=[
            *aug_transforms(size=img_size),
            Normalize.from_stats(*imagenet_stats),
        ],
    )
    return dblock.dataloaders(path, bs=bs)


def cmd_train(data_path: str, epochs: int):
    """Train piece classifier on per-class images."""
    dls = get_dls(Path(data_path))
    learn = vision_learner(dls, models.resnet34, metrics=[accuracy], cbs=[MixUp()])
    lr, _ = learn.lr_find(suggest_funcs=(minimum,))(0)
    learn.fine_tune(
        epochs,
        base_lr=lr,
        cbs=[
            SaveModelCallback(monitor="accuracy", fname="best"),
            EarlyStoppingCallback(monitor="accuracy", patience=3),
        ],
    )
    learn.export("chess_classifier.pkl")
    print("Saved chess_classifier.pkl")


def cmd_infer(input_path: str, model_path: str):
    """Infer FEN from a board PDF or image."""
    img = (
        pdf_to_image(input_path)
        if input_path.lower().endswith(".pdf")
        else cv2.imread(input_path)
    )
    cells = warp_and_segment(img)
    learn = load_learner(model_path).to_fp16()
    preds = [
        [
            str(learn.predict(PILImage.create(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))[0])
            for c in row
        ]
        for row in cells
    ]
    # build FEN
    fen_r = []
    for row in preds:
        f, empty = "", 0
        for c in row:
            if c == "empty":
                empty += 1
            else:
                if empty > 0:
                    f += str(empty)
                    empty = 0
                f += c
        if empty > 0:
            f += str(empty)
        fen_r.append(f)
    fen = "/".join(fen_r) + " w KQkq - 0 1"
    print("FEN:", fen)


# ----- Main -----


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    seg = sub.add_parser("segment")
    seg.add_argument("--boards-dir", required=True)
    seg.add_argument("--out-dir", required=True)
    prp = sub.add_parser("prepare")
    prp.add_argument("--boards-dir", required=True)
    prp.add_argument("--labels-json", required=True)
    prp.add_argument("--out-dir", required=True)
    trn = sub.add_parser("train")
    trn.add_argument("--data-path", required=True)
    trn.add_argument("--epochs", type=int, default=5)
    inf = sub.add_parser("infer")
    inf.add_argument("--input", required=True)
    inf.add_argument("--model", default="chess_classifier.pkl")

    args = p.parse_args()
    if args.cmd == "segment":
        cmd_segment(args.boards_dir, args.out_dir)
    elif args.cmd == "prepare":
        cmd_prepare(args.boards_dir, args.labels_json, args.out_dir)
    elif args.cmd == "train":
        cmd_train(args.data_path, args.epochs)
    elif args.cmd == "infer":
        cmd_infer(args.input, args.model)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
