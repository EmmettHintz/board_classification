#!/usr/bin/env python3
"""
Utility script to check FEN mappings for black pieces.
This helps diagnose why there might not be black piece directories.
"""

import sys
from pathlib import Path

# Mapping from FEN notation to directory names (must match chess_fen_fastai.py)
FEN_TO_LABEL = {
    "P": "P_white", "N": "N_white", "B": "B_white", "R": "R_white", "Q": "Q_white", "K": "K_white",
    "p": "p_black", "n": "n_black", "b": "b_black", "r": "r_black", "q": "q_black", "k": "k_black"
}

def analyze_fen_file(fen_file_path):
    """Analyze a FEN mapping file to check for black pieces."""
    if not Path(fen_file_path).exists():
        print(f"Error: {fen_file_path} does not exist.")
        return
    
    # Define piece symbols
    white_pieces = ["P", "N", "B", "R", "Q", "K"]
    black_pieces = ["p", "n", "b", "r", "q", "k"]
    
    # Counts and tracking
    total_fens = 0
    fens_with_black = 0
    black_piece_counts = {p: 0 for p in black_pieces}
    white_piece_counts = {p: 0 for p in white_pieces}
    boards_with_black = []
    
    # Read the FEN file
    print(f"Analyzing {fen_file_path}...")
    with open(fen_file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or "," not in line:
                continue
                
            total_fens += 1
            board_name, fen = line.split(",", 1)
            board_name = board_name.strip()
            fen = fen.strip()
            
            # Check for black pieces
            has_black = False
            
            # Only look at the board part (before any spaces)
            fen_board = fen.split(" ")[0]
            
            for char in fen_board:
                if char in white_pieces:
                    white_piece_counts[char] += 1
                elif char in black_pieces:
                    black_piece_counts[char] += 1
                    has_black = True
            
            if has_black:
                fens_with_black += 1
                boards_with_black.append(board_name)
    
    # Print results
    print("\n=== FEN Analysis Results ===")
    print(f"Total FEN strings: {total_fens}")
    print(f"FENs with black pieces: {fens_with_black} ({fens_with_black/total_fens*100:.1f}%)")
    
    print("\nBlack piece counts and directory names:")
    for piece, count in black_piece_counts.items():
        directory = FEN_TO_LABEL.get(piece, "unknown")
        print(f"  {piece} → {directory}: {count}")
    
    print("\nWhite piece counts and directory names:")
    for piece, count in white_piece_counts.items():
        directory = FEN_TO_LABEL.get(piece, "unknown")
        print(f"  {piece} → {directory}: {count}")
    
    if fens_with_black == 0:
        print("\n⚠️  WARNING: NO BLACK PIECES FOUND IN ANY FEN STRINGS!")
        print("This is why there are no images in the black piece directories.")
        print("Check your FEN notation - black pieces should use lowercase letters (p, n, b, r, q, k).")
        print("Standard initial position should have: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
    else:
        print("\nBoards with black pieces:")
        for board in boards_with_black[:10]:  # Show first 10
            print(f"  {board}")
        if len(boards_with_black) > 10:
            print(f"  ... and {len(boards_with_black) - 10} more")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_fen_file(sys.argv[1])
    else:
        print("Usage: python check_fen_mappings.py <fen_mapping_file>")
        print("Example: python check_fen_mappings.py fen_mapping.txt") 