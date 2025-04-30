# Chess Board Recognition and FEN Notation Generation
# Authors: Emmett Hintz, Tajveer Singh, Zach Amendola


# This notebook presents an end-to-end solution for:
# 1. Processing chess board diagrams from PDFs
# 2. Segmenting the chess board into 64 squares
# 3. Classifying the chess pieces in each square using transfer learning
# 4. Converting the classifications to FEN notation
#    FEN notation is a standard way to represent a chess board position

# Import necessary libraries
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
from datetime import datetime
import json
import platform
import sys
import io


# Check TensorFlow version and GPU availability
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")


# Set up paths for local environment
current_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(current_dir, "boards")
output_dir = os.path.join(current_dir, "output")

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}")

# Load and Examine Data
# List all PDF files in the folder
try:
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files")
    if pdf_files:
        print("First few files:", pdf_files[:5])
    else:
        print("No PDF files found in the specified directory!")
except Exception as e:
    print(f"Error accessing {input_dir}: {e}")
    pdf_files = []

# List all PDF files in the boards folder
pdf_files = []
if os.path.exists(input_dir):
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files")
    if pdf_files:
        print("First few files:", pdf_files[:5])
    else:
        print("No PDF files found in the specified directory!")
else:
    print(f"Error: Input directory {input_dir} does not exist!")


"""Define Board Processing Functions
These functions will process the PDF board images, detect the chess board,
and segment it into individual squares.
"""


def pdf_to_image(pdf_path, dpi=200):
    """
    Convert the first page of a PDF to a high-resolution image.
    Mac-optimized version uses pdf2image.

    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for conversion (higher is better for chess diagrams)

    Returns:
        NumPy array containing the image
    """
    print(f"  Converting PDF to image... ", end="", flush=True)
    try:
        # For Mac, we need to handle poppler path differently
        if platform.system() == "Darwin":  # macOS
            # First try the standard approach
            try:
                pages = convert_from_path(pdf_path, dpi=dpi)
            except Exception as e:
                print(f"\n  Standard PDF conversion failed: {e}")
                # If homebrew is installed, poppler might be here
                poppler_path = "/opt/homebrew/bin"
                if os.path.exists(poppler_path):
                    print(f"  Trying with poppler path: {poppler_path}")
                    pages = convert_from_path(
                        pdf_path, dpi=dpi, poppler_path=poppler_path
                    )
                else:
                    raise Exception(
                        "Poppler not found. Install with: brew install poppler"
                    )
        else:
            # For other platforms
            pages = convert_from_path(pdf_path, dpi=dpi)

        # Use the first page
        image = np.array(pages[0])
        print("done")
        return image
    except Exception as e:
        print(f"failed: {e}")
        print("If on Mac, make sure poppler is installed: brew install poppler")
        return None


def preprocess_board_image(image):
    """
    Preprocess the chess board image to enhance features and handle messy grid lines.

    Args:
        image: The input chess board image

    Returns:
        Preprocessed image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply contrast enhancement with adaptive CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # Try different thresholding approaches and select the best one
    # 1. Adaptive thresholding
    binary_adaptive = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 2. Otsu's thresholding
    _, binary_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Combine both thresholding methods for better results
    binary = cv2.bitwise_or(binary_adaptive, binary_otsu)
    
    # Morphological operations to clean up the image
    # Create a small kernel for noise removal
    kernel_small = np.ones((2, 2), np.uint8)
    # Create a larger kernel for connecting components
    kernel_large = np.ones((3, 3), np.uint8)
    
    # Remove small noise
    denoised = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
    
    # Connect nearby components that might be broken
    dilated = cv2.dilate(denoised, kernel_large, iterations=1)
    
    # Use closing to fill small holes
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_large)

    return closed, enhanced


def find_board_contour(image):
    """
    Detect the main chess board contour in the image with enhanced robustness
    for messy or skewed boards.

    Args:
        image: Preprocessed binary image

    Returns:
        Contour of the chessboard
    """
    # Find contours in the image
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found in the image")

    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Initialize best_contour with the largest contour
    best_contour = contours[0]
    
    # Look for a contour that's closer to a square/rectangle shape
    # This helps with boards that have grid lines that might create irregular contours
    best_score = float('inf')
    
    # Check several of the largest contours
    for contour in contours[:min(5, len(contours))]:
        # Skip very small contours
        if cv2.contourArea(contour) < 0.3 * cv2.contourArea(contours[0]):
            continue
            
        # Calculate bounding rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)  # Changed from np.int0 to np.int32
        rect_area = cv2.contourArea(box)
        
        # Calculate shape regularity (how close to rectangular)
        contour_area = cv2.contourArea(contour)
        if rect_area > 0:
            area_score = abs(1.0 - contour_area / rect_area)
            
            # Check if the perimeter-to-area ratio is reasonable
            perimeter = cv2.arcLength(contour, True)
            if contour_area > 0:
                compactness = perimeter * perimeter / contour_area
                
                # For a perfect square, this value is about 16
                compactness_score = abs(16 - compactness)
                
                # Combine scores
                score = area_score + 0.1 * compactness_score
                
                if score < best_score:
                    best_score = score
                    best_contour = contour

    # Approximate the contour to get a cleaner polygon
    epsilon = 0.02 * cv2.arcLength(best_contour, True)
    approx_board = cv2.approxPolyDP(best_contour, epsilon, True)
    
    # If our approximation doesn't have 4 points, use the minimum area rectangle
    if len(approx_board) != 4:
        rect = cv2.minAreaRect(best_contour)
        approx_board = cv2.boxPoints(rect)
        approx_board = np.int32(approx_board)  # Changed from np.int0 to np.int32

    return approx_board


def get_board_corners(contour, original_image):
    """
    Get the four corners of the chessboard from its contour.
    Uses the simpler, faster method that was working previously.

    Args:
        contour: Chessboard contour
        original_image: Original image for visualization

    Returns:
        Four corners of the chessboard as a numpy array
    """
    # If we have exactly 4 points this is great, otherwise use different approach
    if len(contour) == 4:
        corners = contour.reshape(4, 2)
    else:
        # Get bounding rectangle as fallback
        rect = cv2.minAreaRect(contour)
        corners = cv2.boxPoints(rect).astype(np.int32)

    # Sort corners to be in the order: top-left, top-right, bottom-right, bottom-left
    # First, compute center of contour
    center = np.mean(corners, axis=0)
    
    # Function to sort corners
    def sort_corners(corners, center):
        # Calculate the angles
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        # Sort by angles
        sorted_indices = np.argsort((-angles + 2.5 * np.pi) % (2 * np.pi))
        sorted_corners = corners[sorted_indices]
        # Ensure the first point is top-left
        if sorted_corners[0][1] > center[1]:  # If y is below center
            sorted_corners = np.roll(sorted_corners, -1, axis=0)
        return sorted_corners
        
    corners = sort_corners(corners, center)
    
    # Debug image with corners marked
    debug_img = original_image.copy()
    for i, corner in enumerate(corners):
        cv2.circle(debug_img, tuple(corner), 15, (0, 255, 0), -1)
        cv2.putText(
            debug_img,
            str(i),
            tuple(corner),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

    return corners, debug_img


def perspective_transform(image, corners):
    """
    Apply perspective transformation to get a top-down view of the chessboard.

    Args:
        image: Original image
        corners: Four corners of the chessboard

    Returns:
        Warped image (top-down view)
    """
    # Desired size of the output
    board_size = 800
    dst_corners = np.array(
        [[0, 0], [board_size, 0], [board_size, board_size], [0, board_size]],
        dtype=np.float32,
    )

    # Get perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(
        corners.astype(np.float32), dst_corners
    )

    # Apply perspective transformation
    warped = cv2.warpPerspective(image, transform_matrix, (board_size, board_size))

    return warped


def segment_chessboard(warped_image):
    """
    Segment the chessboard into 64 individual squares with enhanced robustness
    for messy or uneven grid lines.

    Args:
        warped_image: Warped chessboard image

    Returns:
        8x8 grid of square images
    """
    height, width = warped_image.shape[:2]
    
    # Try to detect actual grid lines
    if len(warped_image.shape) == 3:
        gray = cv2.cvtColor(warped_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = warped_image.copy()
    
    # Apply adaptive thresholding to find grid lines
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Use morphology to enhance horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width//16, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height//16))
    
    # Detect horizontal lines
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=1)
    
    # Detect vertical lines
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=1)
    
    # Combine the lines
    grid_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)
    
    # Function to find the most likely grid line positions
    def find_grid_lines(lines_image, axis=0):
        # For horizontal lines, axis=0 (project onto y-axis)
        # For vertical lines, axis=1 (project onto x-axis)
        projection = np.sum(lines_image, axis=axis)
        
        # Find peaks in the projection (likely grid positions)
        peak_threshold = np.max(projection) * 0.3  # Adjust threshold as needed
        peaks = []
        
        for i in range(1, len(projection) - 1):
            if (projection[i] > projection[i-1] and 
                projection[i] > projection[i+1] and 
                projection[i] > peak_threshold):
                peaks.append(i)
        
        # If we don't have enough peaks, use regular intervals
        if len(peaks) < 9:  # We need 9 grid lines (8 cells + boundaries)
            return np.linspace(0, lines_image.shape[1-axis] - 1, 9).astype(int), projection, []
        
        # Sort peaks and select the most prominent ones
        peaks.sort()
        original_peaks = peaks.copy()
        
        # If too many peaks, select the most evenly spaced ones
        if len(peaks) > 9:
            # Start with the first and last peaks (board boundaries)
            selected_peaks = [peaks[0], peaks[-1]]
            
            # Calculate average spacing
            avg_spacing = (peaks[-1] - peaks[0]) / 8
            
            # Try to find peaks at expected positions
            for i in range(1, 8):
                expected_pos = peaks[0] + i * avg_spacing
                best_peak = None
                min_distance = float('inf')
                
                for peak in peaks[1:-1]:  # Skip first and last which we already selected
                    if peak not in selected_peaks:
                        distance = abs(peak - expected_pos)
                        if distance < min_distance:
                            min_distance = distance
                            best_peak = peak
                
                if best_peak is not None and min_distance < avg_spacing * 0.5:
                    selected_peaks.append(best_peak)
            
            # If we still don't have enough, fill in with evenly spaced positions
            if len(selected_peaks) < 9:
                even_spacing = np.linspace(peaks[0], peaks[-1], 9)
                
                # Add missing positions
                for i in range(1, 8):
                    if i < len(selected_peaks) - 1:
                        continue
                    
                    pos = even_spacing[i]
                    closest_peak = None
                    min_dist = float('inf')
                    
                    for peak in peaks:
                        if peak not in selected_peaks:
                            dist = abs(peak - pos)
                            if dist < min_dist and dist < avg_spacing * 0.4:
                                min_dist = dist
                                closest_peak = peak
                    
                    if closest_peak is not None:
                        selected_peaks.append(closest_peak)
                    else:
                        selected_peaks.append(int(pos))
            
            # Sort the selected peaks
            selected_peaks.sort()
            
            # Ensure we have exactly 9 peaks
            if len(selected_peaks) > 9:
                selected_peaks = selected_peaks[:9]
            elif len(selected_peaks) < 9:
                # Fill remaining positions with evenly spaced points
                missing = 9 - len(selected_peaks)
                evenly_spaced = np.linspace(selected_peaks[0], selected_peaks[-1], 9)
                
                # Find positions that don't have a selected peak nearby
                for pos in evenly_spaced:
                    if len(selected_peaks) >= 9:
                        break
                    
                    # Check if this position is not close to any existing peak
                    if all(abs(pos - peak) > avg_spacing * 0.3 for peak in selected_peaks):
                        selected_peaks.append(int(pos))
                
                # Sort again
                selected_peaks.sort()
                
                # If still not enough, just use evenly spaced positions
                if len(selected_peaks) < 9:
                    selected_peaks = np.linspace(selected_peaks[0], selected_peaks[-1], 9).astype(int)
            
            return np.array(selected_peaks), projection, original_peaks
        
        return np.array(peaks), projection, original_peaks
    
    # Get grid line positions
    row_positions, row_projection, original_row_peaks = find_grid_lines(horizontal_lines, axis=1)
    col_positions, col_projection, original_col_peaks = find_grid_lines(vertical_lines, axis=0)
    
    # Ensure we have exactly 9 positions each (8 squares + boundaries)
    if len(row_positions) != 9 or len(col_positions) != 9:
        # Fall back to regular grid if line detection failed
        row_positions = np.linspace(0, height - 1, 9).astype(int)
        col_positions = np.linspace(0, width - 1, 9).astype(int)
    
    # Create an 8x8 grid to store each square
    squares = []
    for i in range(8):
        squares_row = []
        for j in range(8):
            # Extract the square using detected grid lines
            y_start = row_positions[i]
            y_end = row_positions[i + 1]
            x_start = col_positions[j]
            x_end = col_positions[j + 1]
            
            # Ensure valid boundaries
            y_start = max(0, y_start)
            y_end = min(height, y_end)
            x_start = max(0, x_start)
            x_end = min(width, x_end)
            
            if y_end > y_start and x_end > x_start:
                square = warped_image[y_start:y_end, x_start:x_end]
                # Resize to standardize the square size
                square = cv2.resize(square, (100, 100))
                squares_row.append(square)
            else:
                # If boundaries are invalid, create an empty square
                empty_square = np.zeros((100, 100, 3), dtype=np.uint8) if len(warped_image.shape) == 3 else np.zeros((100, 100), dtype=np.uint8)
                squares_row.append(empty_square)
        
        squares.append(squares_row)
    
    return squares


def process_chessboard(pdf_path, debug=False, timeout=60):
    """
    Complete pipeline to process a chessboard from PDF to segmented squares.

    Args:
        pdf_path: Path to the PDF file
        debug: Whether to show debug images
        timeout: Maximum time in seconds to spend on processing

    Returns:
        8x8 grid of square images
    """
    # Use a simpler progress reporting
    print(f"Processing chessboard from {os.path.basename(pdf_path)}:")
    
    # Convert PDF to image
    image = pdf_to_image(pdf_path)

    if image is None:
        print("  Failed to convert PDF to image")
        return None

    # Preprocess the image
    print("  Preprocessing image... ", end="", flush=True)
    binary, enhanced = preprocess_board_image(image)
    print("done")

    # Find the board contour
    try:
        print("  Finding board contour... ", end="", flush=True)
        board_contour = find_board_contour(binary)
        print("done")
    except ValueError as e:
        print(f"failed: {e}")
        if debug:
            plt.figure(figsize=(10, 10))
            plt.imshow(binary, cmap="gray")
            plt.title("Binary image - no contour found")
            plt.show()
        return None

    # Get the corners of the board
    print("  Detecting board corners... ", end="", flush=True)
    corners, debug_img = get_board_corners(board_contour, image)
    print("done")

    if debug:
        plt.figure(figsize=(10, 10))
        plt.imshow(debug_img)
        plt.title("Detected Corners")
        plt.show()

    # Apply perspective transformation
    print("  Applying perspective transform... ", end="", flush=True)
    warped = perspective_transform(enhanced, corners)
    print("done")

    if debug:
        plt.figure(figsize=(8, 8))
        plt.imshow(warped, cmap="gray")
        plt.title("Warped Chessboard")
        plt.show()
        
    # Segment the chessboard
    print("  Segmenting chessboard into squares... ", end="", flush=True)
    squares = segment_chessboard(warped)
    print("done")

    if debug:
        # Show a sample of squares
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()

        for i in range(16):
            row, col = i // 4, i % 4
            square_img = squares[row][col]
            axes[i].imshow(square_img, cmap="gray")
            axes[i].set_title(f"Square {row},{col}")
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()
    
    print("  Board processing complete")
    return squares


def visualize_board_with_predictions(squares, predictions, class_labels):
    """
    Visualize the chessboard with predictions.

    Args:
        squares: 8x8 grid of square images
        predictions: 8x8 grid of predicted class indices
        class_labels: List of class labels

    Returns:
        Visualization of the board with predictions
    """
    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))

    # Piece symbols for display
    piece_symbols = {
        "empty": "",
        "p": "♙",
        "n": "♘",
        "b": "♗",
        "r": "♖",
        "q": "♕",
        "k": "♔",
        "P": "♟",
        "N": "♞",
        "B": "♝",
        "R": "♜",
        "Q": "♛",
        "K": "♚",
    }

    # Square colors
    light_square = np.ones((3,)) * 240 / 255  # Light color
    dark_square = np.ones((3,)) * 180 / 255  # Dark color

    for row in range(8):
        for col in range(8):
            ax = axes[row, col]
            square_img = squares[row][col]
            prediction = predictions[row][col]

            # Display the square image
            ax.imshow(square_img, cmap="gray")

            # Add colored overlay for the square
            is_light = (row + col) % 2 == 0
            square_color = light_square if is_light else dark_square
            ax.set_facecolor(square_color)

            # Add the predicted piece symbol
            symbol = piece_symbols.get(prediction, "?")
            color = (
                "black" if prediction.isupper() or prediction == "empty" else "white"
            )
            ax.text(
                0.5,
                0.5,
                symbol,
                fontsize=20,
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
            )

            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # Add thin border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(0.5)

    # Add row/column labels (chess notation)
    for i in range(8):
        axes[i, 0].set_ylabel(f"{8-i}", rotation=0, size=12, labelpad=10)
        axes[7, i].set_xlabel(chr(97 + i), size=12)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    return fig


def visualize_detected_board(squares, piece_grid):
    """
    Create a visualization of the board with detected pieces.

    Args:
        squares: 8x8 grid of square images
        piece_grid: 8x8 grid of piece classifications

    Returns:
        None (displays the visualization)
    """
    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))

    # Piece symbols for display
    piece_symbols = {
        "empty": "",
        "p": "♙",
        "n": "♘",
        "b": "♗",
        "r": "♖",
        "q": "♕",
        "k": "♔",
        "P": "♟",
        "N": "♞",
        "B": "♝",
        "R": "♜",
        "Q": "♛",
        "K": "♚",
    }

    # Square colors
    light_square = np.ones((3,)) * 240 / 255  # Light color
    dark_square = np.ones((3,)) * 180 / 255  # Dark color

    for row in range(8):
        for col in range(8):
            ax = axes[row, col]
            square_img = squares[row][col]
            prediction = piece_grid[row][col]

            # Display the square image
            ax.imshow(square_img, cmap="gray")

            # Add colored overlay for the square
            is_light = (row + col) % 2 == 0
            square_color = light_square if is_light else dark_square
            ax.set_facecolor(square_color)

            # Add the predicted piece symbol
            symbol = piece_symbols.get(prediction, "?")
            color = (
                "black" if prediction.isupper() or prediction == "empty" else "white"
            )
            ax.text(
                0.5,
                0.5,
                symbol,
                fontsize=20,
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
            )

            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # Add thin border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(0.5)

    # Add row/column labels (chess notation)
    for i in range(8):
        axes[i, 0].set_ylabel(f"{8-i}", rotation=0, size=12, labelpad=10)
        axes[7, i].set_xlabel(chr(97 + i), size=12)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


# Test the Board Processing Pipeline on one PDF
if pdf_files:
    example_pdf = os.path.join(input_dir, pdf_files[0])
    print(f"Testing board processing on: {pdf_files[0]}")

    # Process the board with debugging enabled
    squares = process_chessboard(example_pdf, debug=True)

    if squares:
        print(
            f"Successfully segmented board into {len(squares)}x{len(squares[0])} squares."
        )
    else:
        print("Failed to process the board. Check the error messages above.")
else:
    print("No PDF files available to test the pipeline.")


# Build Chess Piece Classification Model with Mac Optimization
def build_chess_classification_model(num_classes=13, input_shape=(224, 224, 3)):
    """
    Build an improved transfer learning model using EfficientNetB0.
    Optimized for Mac (MPS or CPU).

    Args:
        num_classes: Number of classes (empty + 12 chess pieces)
        input_shape: Input image dimensions

    Returns:
        Compiled model ready for training
    """
    # For Mac, check if we can use Metal Performance Shaders (MPS)
    if platform.system() == "Darwin" and hasattr(tf.config, "list_physical_devices"):
        # Try to use MPS if available on Mac
        try:
            if len(tf.config.list_physical_devices("GPU")) == 0:
                print("No GPU found, checking for Apple Metal...")
                # Check for macOS 12.0+ devices with Apple Silicon
                if hasattr(tf.config, "list_physical_devices") and hasattr(
                    tf.config, "experimental"
                ):
                    if hasattr(tf.config.experimental, "set_visible_devices"):
                        mps_devices = tf.config.list_physical_devices("MPS")
                        if len(mps_devices) > 0:
                            print(f"MPS device found: {mps_devices}")
                            tf.config.experimental.set_visible_devices(
                                mps_devices[0], "MPS"
                            )
                            print("Using Apple Metal for acceleration")
        except Exception as e:
            print(f"Error setting up MPS: {e}")
            print("Falling back to CPU")

    # Load EfficientNetB0 with pre-trained weights, without the classification layers
    base_model = EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=input_shape
    )

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Build the model
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs)

    # Compile the model - use a lower learning rate for Mac CPU
    lr = (
        0.0005
        if platform.system() == "Darwin"
        and len(tf.config.list_physical_devices("GPU")) == 0
        else 0.001
    )
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def preprocess_input(image):
    """
    Preprocess input images for the model.
    """
    # Standardize pixel values
    image = image.astype("float32") / 255.0

    # Apply contrast enhancement
    image = tf.image.adjust_contrast(image, 1.5)

    return image


def classify_chess_square(square_img, model, class_labels):
    """
    Preprocess and classify a chess square image.

    Args:
        square_img: The image of a chess square
        model: Trained classification model
        class_labels: List of class labels

    Returns:
        Predicted piece label (FEN symbol or "empty")
    """
    # Ensure the image is RGB (convert if grayscale)
    if len(square_img.shape) == 2 or square_img.shape[2] == 1:
        square_rgb = cv2.cvtColor(square_img, cv2.COLOR_GRAY2RGB)
    else:
        square_rgb = square_img.copy()

    # Resize to model input size
    square_resized = cv2.resize(square_rgb, (224, 224))

    # Preprocess
    input_tensor = preprocess_input(square_resized)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # Predict
    pred = model.predict(input_tensor, verbose=0)
    pred_class = np.argmax(pred, axis=1)[0]
    confidence = pred[0][pred_class]

    # Return prediction only if confidence is above threshold, otherwise "uncertain"
    if confidence > 0.7:
        return class_labels[pred_class]
    else:
        # You could implement a fallback strategy here
        return class_labels[pred_class]  # Still return best guess


# Convert Classifications to FEN Notation - chess notation
def convert_board_to_fen(predictions_matrix):
    """
    Convert an 8x8 matrix of piece predictions to FEN notation.

    Args:
        predictions_matrix: 8x8 array of piece labels

    Returns:
        FEN string representation
    """
    fen_rows = []
    for row in predictions_matrix:
        fen_row = ""
        empty_count = 0
        for cell in row:
            if cell == "empty":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += cell  # Assume cell value is already in FEN format
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)

    # FEN notation represents the board from rank 8 to rank 1
    fen = "/".join(fen_rows)

    # Add placeholders for the rest of the FEN notation
    # Full FEN: position + active color + castling + en passant + halfmove + fullmove
    fen += " w KQkq - 0 1"

    return fen


def process_board_with_hybrid_detection(pdf_path, debug=False, output_dir=None):
    """
    Process a chess board image and detect pieces using hybrid approach.
    Enhanced with additional debug visualizations.

    Args:
        pdf_path: Path to the PDF file
        debug: Whether to show debug visualizations
        output_dir: Directory to save debug visualizations (optional)

    Returns:
        FEN notation, board squares for visualization, and piece grid
    """
    # Create output directory for debug visualizations if specified
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get PDF filename for saving visualizations
    if output_dir is not None:
        pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        vis_path = os.path.join(output_dir, f"{pdf_filename}_debug")
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
    else:
        vis_path = None
        
    # Process the board to get squares
    squares = process_chessboard(pdf_path, debug=debug)

    if squares is None:
        return None, None, None
    
    # Save some sample squares if output directory is specified
    if vis_path is not None:
        # Save a sample of squares
        fig, axes = plt.subplots(8, 8, figsize=(20, 20))
        
        for row in range(8):
            for col in range(8):
                square_img = squares[row][col]
                axes[row, col].imshow(square_img, cmap="gray" if len(square_img.shape) == 2 else None)
                axes[row, col].set_title(f"Square {row},{col}")
                axes[row, col].axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_path, "all_squares.png"), dpi=150)
        plt.close(fig)

    # Use hybrid detection to identify pieces
    piece_grid = detect_chess_pieces_hybrid(squares)

    # Convert to FEN notation
    print("  Converting to FEN notation... ", end="", flush=True)
    fen = convert_board_to_fen(piece_grid)
    print("done")
    
    # Save the detected board if output directory is specified
    if vis_path is not None:
        # Create visualization 
        fig = visualize_board_with_predictions(squares, piece_grid, [])
        plt.savefig(os.path.join(vis_path, "detected_board.png"), dpi=150)
        plt.close(fig)
        
        # Save FEN notation
        with open(os.path.join(vis_path, "fen.txt"), "w") as f:
            f.write(fen)

    # Return the original squares and piece grid
    return fen, squares, piece_grid


# Create a direct visualization function
def direct_visualize_board(squares, piece_grid):
    """
    Create a visualization array directly from squares and piece grid.

    Args:
        squares: 8x8 grid of square images
        piece_grid: 8x8 grid of piece classifications

    Returns:
        Numpy array for the visualization
    """
    # Determine square size
    square_height, square_width = squares[0][0].shape[:2]

    # Create empty canvas for the board (8x8 squares)
    board_height = square_height * 8
    board_width = square_width * 8

    # Create RGB image (convert if grayscale)
    if len(squares[0][0].shape) == 2:  # Grayscale
        board_image = np.zeros((board_height, board_width, 3), dtype=np.uint8)
    else:  # Already RGB
        board_image = np.zeros(
            (board_height, board_width, squares[0][0].shape[2]), dtype=np.uint8
        )

    # Piece symbols and colors
    piece_colors = {
        "empty": None,
        "p": (0, 0, 0),  # Black color for white pieces (uppercase)
        "n": (0, 0, 0),
        "b": (0, 0, 0),
        "r": (0, 0, 0),
        "q": (0, 0, 0),
        "k": (0, 0, 0),
        "P": (255, 255, 255),  # White color for black pieces (lowercase)
        "N": (255, 255, 255),
        "B": (255, 255, 255),
        "R": (255, 255, 255),
        "Q": (255, 255, 255),
        "K": (255, 255, 255),
    }

    piece_labels = {
        "empty": "",
        "P": "P",
        "N": "N",
        "B": "B",
        "R": "R",
        "Q": "Q",
        "K": "K",
        "p": "p",
        "n": "n",
        "b": "b",
        "r": "r",
        "q": "q",
        "k": "k",
    }

    # Copy each square image to the board
    for row in range(8):
        for col in range(8):
            # Get square image and convert to RGB if needed
            square_img = squares[row][col]
            if len(square_img.shape) == 2:  # Grayscale
                square_rgb = cv2.cvtColor(square_img, cv2.COLOR_GRAY2RGB)
            else:
                square_rgb = square_img.copy()

            # Calculate position
            y_start = row * square_height
            y_end = (row + 1) * square_height
            x_start = col * square_width
            x_end = (col + 1) * square_width

            # Add square to board
            board_image[y_start:y_end, x_start:x_end] = square_rgb

            # Add piece label overlay
            piece = piece_grid[row][col]
            if piece != "empty":
                # Create label
                label = piece_labels[piece]
                color = piece_colors[piece]

                # Calculate text position
                text_x = x_start + square_width // 2
                text_y = y_start + square_height // 2

                # Add text (scaled based on square size)
                font_scale = square_width / 100
                thickness = max(1, int(square_width / 50))
                cv2.putText(
                    board_image,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

    # Add row/column labels
    font_scale = square_width / 150
    thickness = max(1, int(square_width / 100))
    text_color = (0, 0, 0)  # Black text

    # Add column labels (a-h)
    for col in range(8):
        label = chr(97 + col)  # 'a' through 'h'
        x_pos = col * square_width + square_width // 2
        y_pos = board_height - 10
        cv2.putText(
            board_image,
            label,
            (x_pos, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    # Add row labels (1-8)
    for row in range(8):
        label = str(8 - row)  # '8' through '1'
        x_pos = 10
        y_pos = row * square_height + square_height // 2
        cv2.putText(
            board_image,
            label,
            (x_pos, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    return board_image


def detect_chess_pieces_hybrid(squares):
    """
    A hybrid approach to detect chess pieces using traditional CV techniques
    tailored specifically for the hand-drawn notation in the chess diagrams.

    Args:
        squares: 8x8 grid of square images

    Returns:
        8x8 grid of piece classifications in FEN notation
    """
    print("  Detecting chess pieces... ", end="", flush=True)
    
    # Initialize an 8x8 grid for piece classifications
    piece_grid = [["empty" for _ in range(8)] for _ in range(8)]

    # Function to check if a square contains text/notation
    def has_text(square_img):
        # Convert to grayscale if not already
        if len(square_img.shape) == 3:
            gray = cv2.cvtColor(square_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = square_img.copy()

        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Calculate the percentage of non-zero pixels
        non_zero = np.count_nonzero(binary)
        total = binary.shape[0] * binary.shape[1]
        pixel_percentage = (non_zero / total) * 100

        # Lower threshold to detect even faint markings
        return pixel_percentage > 0.8
        
    # Function to identify the piece based on visible text
    def identify_piece(square_img):
        # Convert to grayscale if not already
        if len(square_img.shape) == 3:
            gray = cv2.cvtColor(square_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = square_img.copy()

        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Crop to center area to minimize grid line interference
        h, w = binary.shape
        crop_margin = int(min(h, w) * 0.15)
        cropped = binary[crop_margin:-crop_margin, crop_margin:-crop_margin]

        # Use connected components to find contours
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cropped)

        # If there are no significant components, it's an empty square
        if num_labels <= 1:
            return "empty"

        # Filter out small noisy components
        valid_components = []
        for i in range(1, num_labels):  # Skip background label (0)
            if stats[i, cv2.CC_STAT_AREA] > 10:  # Lower threshold to catch smaller marks
                valid_components.append(i)

        if not valid_components:
            return "empty"

        # Check for subscript 'w' to determine color
        has_subscript_w = False
        main_components = []
        
        # Sort components by area (largest to smallest)
        valid_components.sort(key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
        
        # First, identify the main component and any potential subscript 'w'
        for i in valid_components:
            # Get component's center y-position relative to image height
            component_top = stats[i, cv2.CC_STAT_TOP]
            component_height = stats[i, cv2.CC_STAT_HEIGHT]
            component_bottom = component_top + component_height
            rel_bottom = component_bottom / cropped.shape[0]
            
            # Get component's dimensions
            comp_width = stats[i, cv2.CC_STAT_WIDTH]
            comp_height = stats[i, cv2.CC_STAT_HEIGHT]
            comp_area = stats[i, cv2.CC_STAT_AREA]
            comp_density = comp_area / (comp_width * comp_height) if comp_width * comp_height > 0 else 0
            
            # If component is in bottom portion and relatively small, it might be a 'w'
            if (rel_bottom > 0.7 and 
                comp_width < cropped.shape[1] / 4 and 
                comp_height < cropped.shape[0] / 4 and
                comp_density > 0.4):  # 'w' tends to have good fill ratio
                has_subscript_w = True
            else:
                main_components.append(i)
        
        # If no main components remain, default to pawn
        if not main_components:
            piece_type = "P"
        else:
            # Use the largest remaining component for piece identification
            main_component = main_components[0]
            
            # Get shape metrics
            width = stats[main_component, cv2.CC_STAT_WIDTH]
            height = stats[main_component, cv2.CC_STAT_HEIGHT]
            area = stats[main_component, cv2.CC_STAT_AREA]
            aspect = width / height if height > 0 else 1.0
            
            # Calculate density (fill ratio)
            density = area / (width * height) if width * height > 0 else 0
            
            # Get relative position
            left = stats[main_component, cv2.CC_STAT_LEFT]
            top = stats[main_component, cv2.CC_STAT_TOP]
            rel_x = (left + width / 2) / cropped.shape[1]  # Relative x center
            rel_y = (top + height / 2) / cropped.shape[0]  # Relative y center
            rel_area = area / (cropped.shape[0] * cropped.shape[1])  # Relative area
            
            # Enhanced piece type determination
            # By default, assume a pawn
            piece_type = "P"
            
            # Determine piece type based on shape characteristics
            if aspect > 1.2:  # Wider than tall
                if rel_area > 0.2:
                    piece_type = "K"  # Kings tend to be larger and wider
                else:
                    piece_type = "R"  # Rooks tend to be wide
            elif aspect < 0.7:  # Taller than wide
                if rel_area < 0.06:
                    piece_type = "P"  # Pawns tend to be small and slender
                elif density > 0.6:
                    piece_type = "Q"  # Queens often have good fill
                else:
                    piece_type = "B"  # Bishops can be tall and thin
            else:  # Moderate aspect ratio
                if rel_area > 0.15:
                    if density < 0.5:
                        piece_type = "N"  # Knights have moderate density
                    else:
                        piece_type = "K"  # Could be a king
                elif rel_area < 0.08:
                    piece_type = "P"  # Small pieces are likely pawns
                else:
                    # Look at position and size for more clues
                    if density > 0.6:
                        piece_type = "Q"  # Queens tend to have high density
                    elif density < 0.4:
                        piece_type = "N"  # Knights have lower density
                    else:
                        piece_type = "B"  # Default to bishop for moderate values
            if has_subscript_w:
                piece_type = piece_type.lower()
            print(f"[{row},{col}] → aspect={aspect:.2f}, area={rel_area:.3f}, density={density:.2f} → predicted={piece_type}")

        #if has_subscript_w:
        #    piece_type = piece_type.lower()
        #print(f"[{row},{col}] → aspect={aspect:.2f}, area={rel_area:.3f}, density={density:.2f} → predicted={piece_type}")
        return piece_type

    # Process each square in the grid
    squares_processed = 0
    
    for row in range(8):
        for col in range(8):
            square_img = squares[row][col]

            if has_text(square_img):
                piece_grid[row][col] = identify_piece(square_img)
            else:
                piece_grid[row][col] = "empty"
                
            # Update progress every 16 squares
            squares_processed += 1
            if squares_processed % 16 == 0:
                print(f"{squares_processed//16}/4... ", end="", flush=True)
    
    print("done")
    return piece_grid


# End-to-End Pipeline - Mac Optimized


def main():
    """
    Main function to run the entire pipeline.
    """
    # Determine if we're running in a notebook or as a script
    running_in_notebook = "ipykernel" in sys.modules

    print("=" * 50)
    print("Chess Board Recognition - Enhanced for Messy Boards")
    print("=" * 50)

    # Check if there are PDF files to process
    if not pdf_files:
        print("No PDF files found in the boards directory.")
        return

    # Create debug directory
    debug_dir = os.path.join(output_dir, "debug")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
        print(f"Created debug directory: {debug_dir}")

    # Set the maximum number of files to process by default
    max_files = 5  # Only process 5 files by default for quicker testing
    
    # Process a single PDF file as a test
    example_pdf = os.path.join(input_dir, pdf_files[0])
    print(f"\nTesting enhanced board processing on: {pdf_files[0]}")

    # Process the board with hybrid detection and debugging
    fen, squares, piece_grid = process_board_with_hybrid_detection(
        example_pdf, 
        debug=running_in_notebook,
        output_dir=debug_dir
    )

    if fen:
        print(f"Generated FEN: {fen}")

        # Save visualization if not in a notebook
        if not running_in_notebook:
            # Create figure
            fig = visualize_board_with_predictions(squares, piece_grid, [])
            # Save to file
            output_file = os.path.join(
                output_dir, f"{os.path.splitext(pdf_files[0])[0]}_visualization.png"
            )
            fig.savefig(output_file, dpi=150)
            plt.close(fig)
            print(f"Saved visualization to {output_file}")
            print(f"Detailed debug information saved to {debug_dir}")
        else:
            # In notebook, display the visualization
            visualize_detected_board(squares, piece_grid)
    else:
        print("Failed to process the board.")

    # Ask if user wants to process additional files
    response = ""
    if not running_in_notebook:
        response = input(f"\nProcess additional files? (y/n/all): ").lower()
    
    process_all = response == "all"
    process_some = response == "y" or running_in_notebook
    
    if process_all or process_some:
        print("\nProcessing additional files...")
        results = {}

        # Determine number of files to process
        files_to_process = pdf_files if process_all else pdf_files[:max_files]
        
        print(f"Will process {len(files_to_process)} files")
        
        for i, pdf_file in enumerate(files_to_process):
            pdf_path = os.path.join(input_dir, pdf_file)
            print(f"Processing {i+1}/{len(files_to_process)}: {pdf_file}")

            try:
                # For each board, create a separate debug subfolder
                pdf_debug_dir = os.path.join(debug_dir, os.path.splitext(pdf_file)[0])
                
                fen, squares, piece_grid = process_board_with_hybrid_detection(
                    pdf_path, output_dir=pdf_debug_dir
                )

                if fen:
                    # Store results
                    results[pdf_file] = {
                        "fen": fen,
                        "timestamp": datetime.now().isoformat(),
                    }

                    # Save visualization
                    fig = visualize_board_with_predictions(squares, piece_grid, [])
                    vis_path = os.path.join(
                        output_dir, f"{os.path.splitext(pdf_file)[0]}_vis.png"
                    )
                    fig.savefig(vis_path, dpi=150)
                    plt.close(fig)

                    # Save FEN to text file
                    fen_path = os.path.join(
                        output_dir, f"{os.path.splitext(pdf_file)[0]}_fen.txt"
                    )
                    with open(fen_path, "w") as f:
                        f.write(fen)
                else:
                    print(f"Failed to process {pdf_file}")
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                import traceback
                traceback.print_exc()
                results[pdf_file] = {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }

        # Save all results to JSON
        results_path = os.path.join(output_dir, "classification_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nAll results saved to {output_dir}")
        print(f"Results summary saved to {results_path}")
        print(f"Detailed debug information saved to {debug_dir}")

        # Provide a summary of processing results
        success_count = sum(1 for result in results.values() if "fen" in result)
        error_count = len(results) - success_count
        
        print(f"\nProcessing Summary:")
        print(f"  Total files processed: {len(results)}")
        print(f"  Successfully processed: {success_count}")
        print(f"  Failed to process: {error_count}")
        
        if error_count > 0:
            print("\nFiles with errors:")
            for pdf_file, result in results.items():
                if "error" in result:
                    print(f"  - {pdf_file}: {result['error']}")


# Run the main function if called directly
if __name__ == "__main__":
    main()


class ChessBoardClassifier:
    """
    End-to-end pipeline for classifying chess boards and converting to FEN notation.
    """

    def __init__(self, model_path=None):
        """
        Initialize the classifier.

        Args:
            model_path: Path to a pretrained model (optional)
        """
        # Class labels (FEN notation)
        self.class_labels = [
            "empty",  # Empty square
            "P",
            "N",
            "B",
            "R",
            "Q",
            "K",  # White pieces
            "p",
            "n",
            "b",
            "r",
            "q",
            "k",  # Black pieces
        ]

        # Load or build the model
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model = tf.keras.models.load_model(model_path)
        else:
            print("Building new model")
            self.model = build_chess_classification_model(
                num_classes=len(self.class_labels)
            )

    def process_pdf(self, pdf_path, debug=False):
        """
        Process a PDF containing a chess board.

        Args:
            pdf_path: Path to the PDF file
            debug: Whether to show debug visualizations

        Returns:
            FEN notation, board visualization, and predictions
        """
        # Process the chessboard to get segmented squares
        squares = process_chessboard(pdf_path, debug=debug)

        if squares is None:
            print(f"Failed to process {pdf_path}")
            return None, None, None

        # Classify each square
        predictions = self._classify_board(squares)

        # Convert predictions to FEN
        fen = convert_board_to_fen(predictions)

        # Visualize the result
        board_vis = visualize_board_with_predictions(
            squares, predictions, self.class_labels
        )

        return fen, board_vis, predictions

    def process_batch(self, pdf_folder, output_folder=None):
        """
        Process a batch of PDF files.

        Args:
            pdf_folder: Folder containing PDF files
            output_folder: Folder to save outputs (optional)

        Returns:
            Dictionary of results
        """
        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder)

        results = {}
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

        print(f"Processing {len(pdf_files)} PDF files from {pdf_folder}")

        for i, pdf_file in enumerate(pdf_files):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            print(f"Processing {i+1}/{len(pdf_files)}: {pdf_file}")

            try:
                fen, board_vis, predictions = self.process_pdf(pdf_path)

                if fen is None:
                    continue

                # Store results
                results[pdf_file] = {
                    "fen": fen,
                    "timestamp": datetime.now().isoformat(),
                }

                # Save visualization if output folder is provided
                if output_folder and board_vis:
                    vis_path = os.path.join(
                        output_folder, f"{os.path.splitext(pdf_file)[0]}_vis.png"
                    )
                    board_vis.savefig(vis_path, dpi=150)
                    plt.close(board_vis)

                    # Save FEN to text file
                    fen_path = os.path.join(
                        output_folder, f"{os.path.splitext(pdf_file)[0]}_fen.txt"
                    )
                    with open(fen_path, "w") as f:
                        f.write(fen)

            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                results[pdf_file] = {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }

        # Save all results to JSON
        if output_folder:
            results_path = os.path.join(output_folder, "classification_results.json")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

        return results

    def _classify_board(self, squares):
        """
        Classify all squares on a chessboard.

        Args:
            squares: 8x8 grid of square images

        Returns:
            8x8 grid of predictions
        """
        predictions = []
        for row in squares:
            row_preds = []
            for square_img in row:
                prediction = classify_chess_square(
                    square_img, self.model, self.class_labels
                )
                row_preds.append(prediction)
            predictions.append(row_preds)

        return predictions


def visualize_components(square_img, save_path=None):
    """
    Debug function to visualize the connected components analysis.
    
    Args:
        square_img: The square image to analyze
        save_path: Optional path to save the visualization
        
    Returns:
        Matplotlib figure with visualization
    """
    # Convert to grayscale if needed
    if len(square_img.shape) == 3:
        gray = cv2.cvtColor(square_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = square_img.copy()
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Crop to center area to minimize grid line interference
    h, w = binary.shape
    crop_margin = int(min(h, w) * 0.15)
    cropped = binary[crop_margin:-crop_margin, crop_margin:-crop_margin]
    
    # Use connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cropped)
    
    # Create a color image for visualization
    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background is black
    
    # Create component visualization
    components_img = colors[labels]
    
    # Create the figure
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    # Original image
    axs[0, 0].imshow(gray, cmap='gray')
    axs[0, 0].set_title('Original Square')
    axs[0, 0].axis('off')
    
    # Binary image
    axs[0, 1].imshow(binary, cmap='gray')
    axs[0, 1].set_title('Binary Image')
    axs[0, 1].axis('off')
    
    # Cropped binary
    axs[1, 0].imshow(cropped, cmap='gray')
    axs[1, 0].set_title('Cropped Binary')
    axs[1, 0].axis('off')
    
    # Colored components
    axs[1, 1].imshow(components_img)
    axs[1, 1].set_title(f'Connected Components: {num_labels-1}')
    
    # Add component stats
    for i in range(1, num_labels):
        x = centroids[i][0]
        y = centroids[i][1]
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        density = area / (width * height) if width * height > 0 else 0
        
        axs[1, 1].text(
            x, y, f"{i}: {area:.0f}\n{width:.0f}x{height:.0f}\nD:{density:.2f}", 
            color='white', fontsize=8, 
            bbox=dict(facecolor='black', alpha=0.5)
        )
    
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        
    return fig
    
    
def debug_piece_detection(square_img):
    """
    Debug function to show the step-by-step piece detection process.
    
    Args:
        square_img: The square image to analyze
        
    Returns:
        Piece classification and visualization
    """
    # Convert to grayscale if not already
    if len(square_img.shape) == 3:
        gray = cv2.cvtColor(square_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = square_img.copy()

    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Crop to center area to minimize grid line interference
    h, w = binary.shape
    crop_margin = int(min(h, w) * 0.15)
    cropped = binary[crop_margin:-crop_margin, crop_margin:-crop_margin]

    # Use connected components to find contours
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cropped)

    # If there are no significant components, it's an empty square
    if num_labels <= 1:
        return "empty", "No components found"

    # Filter out small noisy components
    valid_components = []
    for i in range(1, num_labels):  # Skip background label (0)
        if stats[i, cv2.CC_STAT_AREA] > 10:  # Lower threshold to catch smaller marks
            valid_components.append(i)

    if not valid_components:
        return "empty", "No valid components after filtering"

    # Sort components by area (largest to smallest)
    valid_components.sort(key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
    
    # Check for subscript 'w' to determine color
    has_subscript_w = False
    main_components = []
    subscript_component = None
    
    # Component analysis
    component_details = []
    
    # Find subscript w and main components
    for i in valid_components:
        component_top = stats[i, cv2.CC_STAT_TOP]
        component_height = stats[i, cv2.CC_STAT_HEIGHT]
        component_bottom = component_top + component_height
        rel_bottom = component_bottom / cropped.shape[0]
        
        # Get component's dimensions
        comp_width = stats[i, cv2.CC_STAT_WIDTH]
        comp_height = stats[i, cv2.CC_STAT_HEIGHT]
        comp_area = stats[i, cv2.CC_STAT_AREA]
        comp_density = comp_area / (comp_width * comp_height) if comp_width * comp_height > 0 else 0
        
        # Record component details
        component_details.append({
            "id": i,
            "area": comp_area,
            "width": comp_width,
            "height": comp_height,
            "top": component_top,
            "bottom": component_bottom,
            "rel_bottom": rel_bottom,
            "density": comp_density,
            "is_subscript_w": False
        })
        
        # If component is in bottom portion and relatively small, it might be a 'w'
        if (rel_bottom > 0.7 and 
            comp_width < cropped.shape[1] / 4 and 
            comp_height < cropped.shape[0] / 4 and
            comp_density > 0.4):  # 'w' tends to have good fill ratio
            has_subscript_w = True
            subscript_component = i
            component_details[-1]["is_subscript_w"] = True
        else:
            main_components.append(i)
    
    # If no main components remain, default to pawn
    if not main_components:
        return "P" if has_subscript_w else "p", "No main components, defaulting to pawn"
    
    # Use the largest remaining component for piece identification
    main_component = main_components[0]
    
    # Get shape metrics
    width = stats[main_component, cv2.CC_STAT_WIDTH]
    height = stats[main_component, cv2.CC_STAT_HEIGHT]
    area = stats[main_component, cv2.CC_STAT_AREA]
    aspect = width / height if height > 0 else 1.0
    
    # Calculate density (fill ratio)
    density = area / (width * height) if width * height > 0 else 0
    
    # Get relative position
    left = stats[main_component, cv2.CC_STAT_LEFT]
    top = stats[main_component, cv2.CC_STAT_TOP]
    rel_x = (left + width / 2) / cropped.shape[1]  # Relative x center
    rel_y = (top + height / 2) / cropped.shape[0]  # Relative y center
    rel_area = area / (cropped.shape[0] * cropped.shape[1])  # Relative area
    
    # By default, assume a pawn
    piece_type = "P"
    reason = ""
    
    # Determine piece type based on shape characteristics
    if aspect > 1.2:  # Wider than tall
        if rel_area > 0.2:
            piece_type = "K"  # Kings tend to be larger and wider
            reason = f"Wider shape (aspect={aspect:.2f}) with large area (rel_area={rel_area:.2f})"
        else:
            piece_type = "R"  # Rooks tend to be wide
            reason = f"Wider shape (aspect={aspect:.2f}) with moderate area (rel_area={rel_area:.2f})"
    elif aspect < 0.7:  # Taller than wide
        if rel_area < 0.06:
            piece_type = "P"  # Pawns tend to be small and slender
            reason = f"Tall, slender shape (aspect={aspect:.2f}) with small area (rel_area={rel_area:.2f})"
        elif density > 0.6:
            piece_type = "Q"  # Queens often have good fill
            reason = f"Tall shape (aspect={aspect:.2f}) with high density (density={density:.2f})"
        else:
            piece_type = "B"  # Bishops can be tall and thin
            reason = f"Tall shape (aspect={aspect:.2f}) with moderate density (density={density:.2f})"
    else:  # Moderate aspect ratio
        if rel_area > 0.15:
            if density < 0.5:
                piece_type = "N"  # Knights have moderate density
                reason = f"Moderate shape (aspect={aspect:.2f}) with large area (rel_area={rel_area:.2f}) and lower density (density={density:.2f})"
            else:
                piece_type = "K"  # Could be a king
                reason = f"Moderate shape (aspect={aspect:.2f}) with large area (rel_area={rel_area:.2f}) and higher density (density={density:.2f})"
        elif rel_area < 0.08:
            piece_type = "P"  # Small pieces are likely pawns
            reason = f"Moderate shape (aspect={aspect:.2f}) with small area (rel_area={rel_area:.2f})"
        else:
            # Look at position and size for more clues
            if density > 0.6:
                piece_type = "Q"  # Queens tend to have high density
                reason = f"Moderate shape (aspect={aspect:.2f}) with high density (density={density:.2f})"
            elif density < 0.4:
                piece_type = "N"  # Knights have lower density
                reason = f"Moderate shape (aspect={aspect:.2f}) with lower density (density={density:.2f})"
            else:
                piece_type = "B"  # Default to bishop for moderate values
                reason = f"Moderate shape (aspect={aspect:.2f}) with moderate density (density={density:.2f})"
    
    # Add shape metrics to the reason
    stats_info = (f"Main component: Area={area}, Size={width}x{height}, "
                 f"Aspect={aspect:.2f}, Density={density:.2f}, "
                 f"Rel. Area={rel_area:.2f}, Pos=({rel_x:.2f},{rel_y:.2f})")
    
    # Create a detailed analysis
    analysis = {
        "num_components": len(valid_components),
        "has_subscript_w": has_subscript_w,
        "main_component": main_component,
        "subscript_component": subscript_component,
        "metrics": {
            "aspect": aspect,
            "density": density,
            "rel_area": rel_area,
            "rel_x": rel_x,
            "rel_y": rel_y
        },
        "components": component_details,
        "reason": reason,
        "stats": stats_info
    }
    
    # Apply color based on whether we found a subscript 'w'
    return piece_type if has_subscript_w else piece_type.lower(), analysis


def visualize_grid_detection(warped_image, output_path=None):
    """
    Visualize the grid line detection process to help with debugging.
    
    Args:
        warped_image: The warped chessboard image
        output_path: Path to save the visualization (optional)
        
    Returns:
        Visualization figure
    """
    height, width = warped_image.shape[:2]
    
    # Convert to grayscale if needed
    if len(warped_image.shape) == 3:
        gray = cv2.cvtColor(warped_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = warped_image.copy()
    
    # Apply adaptive thresholding to find grid lines
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Use morphology to enhance horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width//16, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height//16))
    
    # Detect horizontal lines
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=1)
    
    # Detect vertical lines
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=1)
    
    # Combine the lines
    grid_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)
    
    # Function to find the most likely grid line positions
    def find_grid_lines(lines_image, axis=0):
        # For horizontal lines, axis=0 (project onto y-axis)
        # For vertical lines, axis=1 (project onto x-axis)
        projection = np.sum(lines_image, axis=axis)
        
        # Find peaks in the projection (likely grid positions)
        peak_threshold = np.max(projection) * 0.3  # Adjust threshold as needed
        peaks = []
        
        for i in range(1, len(projection) - 1):
            if (projection[i] > projection[i-1] and 
                projection[i] > projection[i+1] and 
                projection[i] > peak_threshold):
                peaks.append(i)
        
        # If we don't have enough peaks, use regular intervals
        if len(peaks) < 9:  # We need 9 grid lines (8 cells + boundaries)
            return np.linspace(0, lines_image.shape[1-axis] - 1, 9).astype(int), projection, []
        
        # Sort peaks and select the most prominent ones
        peaks.sort()
        original_peaks = peaks.copy()
        
        # If too many peaks, select the most evenly spaced ones
        if len(peaks) > 9:
            # Start with the first and last peaks (board boundaries)
            selected_peaks = [peaks[0], peaks[-1]]
            
            # Calculate average spacing
            avg_spacing = (peaks[-1] - peaks[0]) / 8
            
            # Try to find peaks at expected positions
            for i in range(1, 8):
                expected_pos = peaks[0] + i * avg_spacing
                best_peak = None
                min_distance = float('inf')
                
                for peak in peaks[1:-1]:  # Skip first and last which we already selected
                    if peak not in selected_peaks:
                        distance = abs(peak - expected_pos)
                        if distance < min_distance:
                            min_distance = distance
                            best_peak = peak
                
                if best_peak is not None and min_distance < avg_spacing * 0.5:
                    selected_peaks.append(best_peak)
            
            # If we still don't have enough, fill in with evenly spaced positions
            if len(selected_peaks) < 9:
                even_spacing = np.linspace(peaks[0], peaks[-1], 9)
                
                # Add missing positions
                for i in range(1, 8):
                    if i < len(selected_peaks) - 1:
                        continue
                    
                    pos = even_spacing[i]
                    closest_peak = None
                    min_dist = float('inf')
                    
                    for peak in peaks:
                        if peak not in selected_peaks:
                            dist = abs(peak - pos)
                            if dist < min_dist and dist < avg_spacing * 0.4:
                                min_dist = dist
                                closest_peak = peak
                    
                    if closest_peak is not None:
                        selected_peaks.append(closest_peak)
                    else:
                        selected_peaks.append(int(pos))
            
            # Sort the selected peaks
            selected_peaks.sort()
            
            # Ensure we have exactly 9 peaks
            if len(selected_peaks) > 9:
                selected_peaks = selected_peaks[:9]
            elif len(selected_peaks) < 9:
                # Fill remaining positions with evenly spaced points
                missing = 9 - len(selected_peaks)
                evenly_spaced = np.linspace(selected_peaks[0], selected_peaks[-1], 9)
                
                # Find positions that don't have a selected peak nearby
                for pos in evenly_spaced:
                    if len(selected_peaks) >= 9:
                        break
                    
                    # Check if this position is not close to any existing peak
                    if all(abs(pos - peak) > avg_spacing * 0.3 for peak in selected_peaks):
                        selected_peaks.append(int(pos))
                
                # Sort again
                selected_peaks.sort()
                
                # If still not enough, just use evenly spaced positions
                if len(selected_peaks) < 9:
                    selected_peaks = np.linspace(selected_peaks[0], selected_peaks[-1], 9).astype(int)
            
            return np.array(selected_peaks), projection, original_peaks
        
        return np.array(peaks), projection, original_peaks
    
    # Get grid line positions
    row_positions, row_projection, original_row_peaks = find_grid_lines(horizontal_lines, axis=1)
    col_positions, col_projection, original_col_peaks = find_grid_lines(vertical_lines, axis=0)
    
    # Ensure we have exactly 9 positions each (8 squares + boundaries)
    if len(row_positions) != 9 or len(col_positions) != 9:
        # Fall back to regular grid if line detection failed
        row_positions = np.linspace(0, height - 1, 9).astype(int)
        col_positions = np.linspace(0, width - 1, 9).astype(int)
    
    # Visualize the results
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    
    # Original image
    axs[0, 0].imshow(warped_image, cmap='gray' if len(warped_image.shape) == 2 else None)
    axs[0, 0].set_title('Original Warped Image')
    axs[0, 0].axis('off')
    
    # Thresholded image
    axs[0, 1].imshow(thresh, cmap='gray')
    axs[0, 1].set_title('Thresholded Image')
    axs[0, 1].axis('off')
    
    # Horizontal lines
    axs[0, 2].imshow(horizontal_lines, cmap='gray')
    axs[0, 2].set_title('Horizontal Lines')
    axs[0, 2].axis('off')
    
    # Vertical lines
    axs[1, 0].imshow(vertical_lines, cmap='gray')
    axs[1, 0].set_title('Vertical Lines')
    axs[1, 0].axis('off')
    
    # Combined grid lines
    axs[1, 1].imshow(grid_lines, cmap='gray')
    axs[1, 1].set_title('Combined Grid Lines')
    axs[1, 1].axis('off')
    
    # Original image with detected grid lines
    grid_overlay = warped_image.copy()
    if len(grid_overlay.shape) == 2:
        grid_overlay = cv2.cvtColor(grid_overlay, cv2.COLOR_GRAY2RGB)
    
    # Draw grid lines
    for row_pos in row_positions:
        cv2.line(grid_overlay, (0, row_pos), (width, row_pos), (0, 255, 0), 2)
    
    for col_pos in col_positions:
        cv2.line(grid_overlay, (col_pos, 0), (col_pos, height), (0, 255, 0), 2)
    
    axs[1, 2].imshow(grid_overlay)
    axs[1, 2].set_title('Detected Grid Lines')
    axs[1, 2].axis('off')
    
    # Plot horizontal line projection
    axs[2, 0].plot(row_projection)
    axs[2, 0].set_title('Horizontal Line Projection')
    
    # Mark detected row positions
    for pos in row_positions:
        axs[2, 0].axvline(x=pos, color='g', linestyle='-', alpha=0.5)
    
    # Mark original peaks
    for pos in original_row_peaks:
        axs[2, 0].axvline(x=pos, color='r', linestyle='--', alpha=0.3)
    
    # Plot vertical line projection
    axs[2, 1].plot(col_projection)
    axs[2, 1].set_title('Vertical Line Projection')
    
    # Mark detected column positions
    for pos in col_positions:
        axs[2, 1].axvline(x=pos, color='g', linestyle='-', alpha=0.5)
    
    # Mark original peaks
    for pos in original_col_peaks:
        axs[2, 1].axvline(x=pos, color='r', linestyle='--', alpha=0.3)
    
    # Show segmented squares sample
    combined_img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Create an 8x8 grid to store each square
    squares = []
    for i in range(8):
        for j in range(8):
            # Extract the square using detected grid lines
            y_start = row_positions[i]
            y_end = row_positions[i + 1]
            x_start = col_positions[j]
            x_end = col_positions[j + 1]
            
            # Ensure valid boundaries
            y_start = max(0, y_start)
            y_end = min(height, y_end)
            x_start = max(0, x_start)
            x_end = min(width, x_end)
            
            if y_end > y_start and x_end > x_start:
                square = warped_image[y_start:y_end, x_start:x_end]
                # Resize to standardize the square size
                square = cv2.resize(square, (50, 50))
                
                # Copy to the combined image
                if len(square.shape) == 2:
                    square_rgb = cv2.cvtColor(square, cv2.COLOR_GRAY2RGB)
                else:
                    square_rgb = square.copy()
                
                r_start = (i * 50)
                r_end = ((i + 1) * 50)
                c_start = (j * 50)
                c_end = ((j + 1) * 50)
                
                if r_end <= 400 and c_end <= 400:
                    combined_img[r_start:r_end, c_start:c_end] = square_rgb
    
    axs[2, 2].imshow(combined_img)
    axs[2, 2].set_title('Segmented Squares Sample')
    axs[2, 2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
    
    return fig
