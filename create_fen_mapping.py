import os
import re
from pathlib import Path


def create_fen_mapping_file(boards_dir, mapping_file, sample_fens=None):
    """
    Create a template FEN mapping file for board PDFs.

    Args:
        boards_dir: Directory with board PDFs
        mapping_file: Output mapping file
        sample_fens: Optional dictionary of sample FEN strings
    """
    # Find all PDF files in the directory
    pdf_files = list(Path(boards_dir).glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {boards_dir}")
        return

    # Extract board names
    board_names = [pdf.stem for pdf in pdf_files]

    # Sort board names naturally (so board10 comes after board9, not board1)
    def natural_sort_key(s):
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]

    board_names.sort(key=natural_sort_key)

    # Write the mapping file
    with open(mapping_file, "w") as f:
        for board_name in board_names:
            # If we have a sample FEN for this board, use it
            if sample_fens and board_name in sample_fens:
                f.write(f"{board_name},{sample_fens[board_name]}\n")
            else:
                # Otherwise just write the board name with a placeholder
                f.write(f"{board_name},\n")

    print(f"Created FEN mapping template at {mapping_file}")
    print(f"Found {len(board_names)} board PDFs")
    print(f"Please fill in the FEN strings for each board and save the file.")


if __name__ == "__main__":
    # Define the directories
    boards_dir = "./input/boards"
    mapping_file = "./fen_mapping.txt"

    # Sample FEN strings for demonstration
    sample_fens = {
        "board1": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
        "board2": "rn1qkb1r/pp2pppp/2p2n2/3p4/3P1B2/4Pb1P/PPP2PP1/RN1QKB1R",
    }

    # Create the mapping file
    create_fen_mapping_file(boards_dir, mapping_file, sample_fens)
