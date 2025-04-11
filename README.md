# Chess Board Recognition
- Emmett Hintz, Tajveer Singh, Zach Amendola

The goal of this project is to provide an end-to-end pipeline for:
1. Processing chess board diagrams from PDFs
2. Segmenting the chess board into 64 squares
3. Classifying the chess pieces in each square using transfer learning
4. Converting the classifications to FEN notation

Work done so far:
1. Drawing chess boards with pieces represented by their first letter and a subscript to denote piece color
    - Zach and Tajveer (50 boards drawn each)
2. Scanning and uploading chess boards
    - Zach and Tajveer (50 boards scanned each)
3. Implemented initial pipeline with proper processing of the board
   - Emmett
      - The pipeline includes:
         - Conversion of chessboard diagrams from PDFs to high-resolution images.
         - Preprocessing of images to enhance features and handle messy grid lines.
         - Detection of the chessboard contour and segmentation into 64 squares.
         - Classification of chess pieces using a hybrid approach combining traditional computer vision techniques and transfer learning.
         - Conversion of the classified board into FEN notation for chess representation.
         - Beginning implementations of debugging.
       - The pipeline needs to be tuned as a lot of the peices are being put on the borders of the chess board at the moment.

## Example of Working Board Contour Detection
![example_board](output/board1_vis.png)
