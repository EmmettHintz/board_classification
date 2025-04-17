# Chess Position Auto-Labeling System

This system allows you to automatically label chess board positions using FEN notation, eliminating the need for manual labeling of individual cells.

## Step 1: Create a FEN Mapping File

First, create a mapping file that associates each PDF board file with its FEN notation:

```bash
python create_fen_mapping.py
```

This will create a template file called `fen_mapping.txt` with entries like:

```
board1,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR
board2,rn1qkb1r/pp2pppp/2p2n2/3p4/3P1B2/4Pb1P/PPP2PP1/RN1QKB1R
board3,
```

Fill in the missing FEN strings for each board. You've already started with these examples:

```
board1,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR
board2,rn1qkb1r/pp2pppp/2p2n2/3p4/3P1B2/4Pb1P/PPP2PP1/RN1QKB1R
board3,r2q1rk1/pp1n1ppp/2p1pn2/3p4/3P1P2/3B1Q1P/PPPN1PP1/R4RK1
board4,2r2rk1/pp3ppp/1q2pn2/3p4/5P2/2PnQN1P/PP3PP1/1R3RK1
board5,2rr2k1/pp4pp/4qp2/3p4/4n3/2PQN2P/PP3PP1/1R3RK1
board6,3r2k1/pp1r2pp/5p2/3N4/n1P1q3/2Q4P/PP3PP1/3R1RK1
```

## Step 2: Run the Auto-Labeling Script

Once your mapping file is complete, run the auto-labeling script:

```bash
python chess_fen_labeler.py --boards-dir ./input/boards --out-dir ./output/classified_cells --fen-mapping ./fen_mapping.txt
```

This will:
1. Load your FEN mappings
2. Segment each board into 64 individual cells
3. Determine the correct piece label for each cell based on the FEN
4. Save each cell image to the appropriate class directory (empty, P, N, B, etc.)

## Step 3: Train Your Model

After auto-labeling is complete, you can train your model using:

```bash
python chess_fen_fastai.py train --data-path ./output/classified_cells --epochs 5
```

## Step 4: Use Your Model to Recognize New Boards

Once trained, you can use your model to recognize new chess boards:

```bash
python chess_fen_fastai.py infer --input path/to/new/board.pdf --model chess_classifier.pkl
```

## Tips for FEN Notation

- FEN notation encodes the entire chess board position in a single string
- Each rank (row) is described from the 8th rank (top) to the 1st rank (bottom)
- Capital letters represent white pieces: PNBRQK
- Lowercase letters represent black pieces: pnbrqk
- Numbers represent consecutive empty squares
- Ranks are separated by forward slashes "/"

For example, the starting position in chess is:
```
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR
```

## Benefits of This Approach

- No need to manually label thousands of individual cell images
- More accurate and consistent labeling
- Significant time savings
- Easy to expand your dataset - just add more board PDFs and their FENs