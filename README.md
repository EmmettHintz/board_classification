# Chess Board Recognition
- Emmett Hintz, Tajveer Singh, Zach Amendola

This project provides an end-to-end pipeline for:
1. Processing chess board diagrams from PDFs
2. Segmenting the chess board into 64 squares
3. Classifying the chess pieces in each square using transfer learning
4. Converting the classifications to FEN notation

## Mac-Specific Optimizations

This version includes several optimizations specifically for macOS:

1. **PDF Processing**: Uses proper Poppler path detection for Mac via Homebrew
2. **TensorFlow Acceleration**: Automatically detects and uses Metal Performance Shaders (MPS) on Apple Silicon 
3. **Dependency Management**: Includes Mac-specific dependency checks and installation instructions
4. **Path Handling**: Uses proper path resolution for Mac file system
5. **Performance Tuning**: Adjusted learning rates and batch sizes for optimal Mac performance

## Installation for Mac

### Prerequisites
- Python 3.8 or higher
- Homebrew package manager (recommended)

### Step 1: Install Homebrew (if not already installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 2: Install Poppler (required for PDF processing)
```bash
brew install poppler
```

### Step 3: Set up Python environment (recommended)
```bash
# Create a virtual environment
python3.10 -m venv venv

# Activate the environment
source venv/bin/activate
```

### Step 4: Install Python dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Place your chess board PDFs in the 'boards' folder

### Run the script
```bash
python board_classification.py
```

### Output
The script will:
1. Process the chess board PDFs
2. Generate FEN notation for each board
3. Create visualizations of the detected pieces
4. Save results to the 'output' folder:
   - `[board_name]_vis.png`: Visual representation of the detected pieces
   - `[board_name]_fen.txt`: FEN notation for the board
   - `classification_results.json`: Summary of all processed boards

## Troubleshooting for Mac

### Apple Silicon (M1/M2/M3) Macs
For optimal performance on Apple Silicon Macs:
```bash
pip install tensorflow-metal
```

### PDF Conversion Issues
If you encounter PDF conversion issues:
1. Ensure Poppler is installed: `brew install poppler`
2. Check if the Poppler binaries are in your PATH: `which pdftoppm`
3. If needed, specify the Poppler path manually in the code

### Performance Optimization
- For better performance, consider reducing the DPI value in the `pdf_to_image` function
- If processing large batches, consider increasing batch size for model inference
