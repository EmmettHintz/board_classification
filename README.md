# Chess Position Recognition System

## Overview

This project presents an end-to-end pipeline for recognizing chess positions from images of hand-drawn chess boards. Our aim is to automate the process of segmenting a chess board, classifying each square, and reconstructing the board's FEN (Forsyth-Edwards Notation) from a single image. We curated a dataset of 95 hand-drawn chess boards, each labeled with its FEN, and developed a system that segments each board, auto-labels the squares, trains a fastai-based classifier, and accurately predicts FEN for new boards. Our main finding is that with a modest, carefully-labeled dataset, it is possible to train a model that generalizes well to new, hand-drawn chess boards.

## Setup

### 1. Prerequisites

- Python 3.10
- [Homebrew](https://brew.sh/) (Mac, recommended)

### 2. Install Poppler (for PDF processing)

- **Mac:**  
  brew install poppler
- **Linux:**  
  sudo apt-get install poppler-utils

### 3. Set Up Python Environment

```bash
python3.10 -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate   # On Windows
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Data Preparation

- The `boards/` directory contains 95 hand-drawn chess board PDFs (`board1.pdf`, ..., `board95.pdf`).
- The `fen_mapping.txt` file maps each board to its FEN string.
- No further data preparation is needed as this is all that is required to segement the board and train the subsequent model using these segmented squares of the chess board.

## Future Directions

There are several promising directions for future work. Expanding the dataset with more diverse hand-drawn boards, including boards with different artistic styles or lighting conditions, would improve generalization. Incorporating data augmentation and more advanced segmentation techniques could further boost accuracy. Finally, extending the system to provide move suggestions from that gamestate could be an interesting and practical application that would be useful for chess education and analysis.

## Contributions

- Emmett Hintz: Pipeline design, code implementation, and documentation (~30 hours)
- Tajveer Singh: Dataset creation and labeling, evaluation metrics (~8 hours)
- Zach Amendola: Dataset creation and labeling, & evaluation of the model (~12 hours)

All members contributed to testing and writing of the final report and poster.

## Features

- PDF chess diagram processing and segmentation
- Automatic labeling based on FEN notation
- Chess piece classifier training using fastai
- FEN notation prediction from new chess board images
- Board-level train/validation splitting for proper evaluation

## Usage

The main script `chess_fen_fastai.py` provides four primary commands:

### 1. Segment and Label Chess Boards

First, create a FEN mapping file that associates each PDF board file with its FEN notation:

```
board1,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR
board2,rn1qkb1r/pp2pppp/2p2n2/3p4/3P1B2/4Pb1P/PPP2PP1/RN1QKB1R
...
```

The provided `fen_mapping.txt` already contains mappings for 95 board positions.

Then process the board PDFs:

```bash
python chess_fen_fastai.py segment --boards-dir ./boards --out-dir ./output --fen-mapping ./fen_mapping.txt
```

This will:

- Load your FEN mappings (labels)
- Segment each board into 64 individual cells
- Determine the correct piece label for each cell based on the FEN
- Save each cell image to the appropriate class directory

### 2. Split Data for Train/Validation

To ensure proper evaluation, split the segmented data at the board level (not image level):

```bash
python chess_fen_fastai.py split --data-path ./output
```

This will:

- Create `./output/split/train` and `./output/split/valid` directories
- Randomly assign 80% of boards to training and 20% to validation
- Ensure all 64 squares from each board go to the same split
- Maintain the same class structure in both splits

You can adjust the validation percentage with `--valid-pct` (default: 0.2).

### 3. Train Your Model

After segmentation and splitting, train your model:

```bash
# Training with the board-level split (recommended)
python chess_fen_fastai.py train --data-path ./output/split --use-split --epochs 5
```

Options:

- `--device cpu` - Force CPU (recommended for M-series Macs with MPS issues)
- `--batch-size` - Adjust batch size (default: 16)
- `--arch` - Choose architecture (default: resnet34)
- `--use-split` - Use the existing train/valid directory structure

### 4. Predict FEN Notation

Once trained, predict FEN notation for new chess boards:

```bash
python chess_fen_fastai.py predict --image path/to/board.pdf
```

The output includes:

- Square-by-square predictions with confidence scores
- The final FEN notation for the entire board

### 5. Evaluate The Model

Evaluate the model's performance on a test set:

```bash
# Evaluate on all boards
python evaluate_fen.py --boards-dir boards/ --fen-mapping fen_mapping.txt --model models/chess_piece_model.pkl --verbose

# OR, evaluate only on validation boards
python evaluate_fen.py --boards-dir boards/ --fen-mapping fen_mapping.txt --model models/chess_piece_model.pkl --split-path output/split --verbose
```

This will:

- Load the model and FEN mapping
- Evaluate the model on the test set (or validation set if using `--split-path`)
- Print various evaluation metrics
  - Exact match
  - Levenshtein average
  - ROGUE-1 precision, recall, f1
  - Square-level accuracy
- Display a bar chart of the model's performance on these metrics

#### Note on Test Set Evaluation:

Since this implementation does not tune hyperparameters based on validation performance, the validation set effectively serves as a true test set. Therefore:

1. For proper evaluation on unseen data, use:

   ```bash
   python evaluate_fen.py --boards-dir boards/ --fen-mapping fen_mapping.txt --model models/chess_piece_model.pkl --split-path output/split --verbose
   ```

2. For an even more rigorous evaluation approach, you can:
   - Set aside some boards before segmentation (e.g., boards 90-95)
   - Only segment and train on the remaining boards
   - Evaluate on these completely held-out boards

The first approach is sufficient for most purposes since our training pipeline does not optimize based on validation performance.

## FEN Notation Explained

FEN (Forsyth-Edwards Notation) is a standard for describing chess positions:

- Each rank (row) is described from the 8th rank (top) to the 1st rank (bottom)
- Capital letters represent white pieces: PNBRQK (Pawn, kNight, Bishop, Rook, Queen, King)
- Lowercase letters represent black pieces: pnbrqk
- Numbers represent consecutive empty squares
- Ranks are separated by forward slashes "/"

Example (starting position): `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR`

## Troubleshooting

### Apple Silicon (M1/M2/M3) Macs

If you encounter MPS errors on Apple Silicon:

1. The script includes PyTorch MPS fallback: `os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"`
2. Use `--device cpu` to force CPU if still having issues

### PDF Conversion Issues

If you encounter PDF conversion issues:

1. Ensure Poppler is installed: `brew install poppler`
2. Check if the Poppler binaries are in your PATH: `which pdftoppm`
3. Try reinstalling pdf2image: `pip install --force-reinstall pdf2image`

### Segmentation Issues

If segmentation is not properly detecting the chess board:

1. Try with a different board PDF
2. Ensure the board has clear borders and contrast

### Data Splitting Issues

If you encounter issues with the data splitting:

1. Make sure the segmentation step has completed successfully
2. Check that your image filenames follow the pattern `boardname_i_j.png`
3. You may need to create the output directory manually: `mkdir -p output`
