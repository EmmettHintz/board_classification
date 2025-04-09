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


def pdf_to_image(pdf_path, dpi=300):
    """
    Convert the first page of a PDF to a high-resolution image.
    Mac-optimized version uses pdf2image.

    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for conversion (higher is better for chess diagrams)

    Returns:
        NumPy array containing the image
    """
    try:
        # For Mac, we need to handle poppler path differently
        if platform.system() == "Darwin":  # macOS
            # First try the standard approach
            try:
                pages = convert_from_path(pdf_path, dpi=dpi)
            except Exception as e:
                print(f"Standard PDF conversion failed: {e}")
                # If homebrew is installed, poppler might be here
                poppler_path = "/opt/homebrew/bin"
                if os.path.exists(poppler_path):
                    print(f"Trying with poppler path: {poppler_path}")
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
        return image
    except Exception as e:
        print(f"Error converting PDF to image: {e}")
        print("If on Mac, make sure poppler is installed: brew install poppler")
        return None


def preprocess_board_image(image):
    """
    Preprocess the chess board image to enhance features.

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

    # Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply adaptive thresholding to handle varying lighting conditions
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Dilate to connect components
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)

    return dilated, enhanced


def find_board_contour(image):
    """
    Detect the main chess board contour in the image.

    Args:
        image: Preprocessed binary image

    Returns:
        Contour of the chessboard
    """
    # Find contours in the image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area in descending order and get the largest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if not contours:
        raise ValueError("No contours found in the image")

    # The largest contour is likely the chessboard
    board_contour = contours[0]

    # Approximate the contour to get a cleaner polygon
    epsilon = 0.02 * cv2.arcLength(board_contour, True)
    approx_board = cv2.approxPolyDP(board_contour, epsilon, True)

    return approx_board


def get_board_corners(contour, original_image):
    """
    Get the four corners of the chessboard from its contour.

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

    # Optional: Debug image with corners marked -- remove if this gets annoying
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
    Segment the chessboard into 64 individual squares.

    Args:
        warped_image: Warped chessboard image

    Returns:
        8x8 grid of square images
    """
    height, width = warped_image.shape[:2]
    square_size = height // 8

    # Create an 8x8 grid to store each square
    squares = []
    for row in range(8):
        squares_row = []
        for col in range(8):
            # Extract the square
            y_start = row * square_size
            y_end = (row + 1) * square_size
            x_start = col * square_size
            x_end = (col + 1) * square_size

            square = warped_image[y_start:y_end, x_start:x_end]
            squares_row.append(square)
        squares.append(squares_row)

    return squares


def process_chessboard(pdf_path, debug=False):
    """
    Complete pipeline to process a chessboard from PDF to segmented squares.

    Args:
        pdf_path: Path to the PDF file
        debug: Whether to show debug images

    Returns:
        8x8 grid of square images
    """
    # Convert PDF to image
    image = pdf_to_image(pdf_path)

    if image is None:
        print(f"Failed to convert PDF to image: {pdf_path}")
        return None

    # Preprocess the image
    binary, enhanced = preprocess_board_image(image)

    # Find the board contour
    try:
        board_contour = find_board_contour(binary)
    except ValueError as e:
        print(f"Error processing {pdf_path}: {e}")
        if debug:
            plt.figure(figsize=(10, 10))
            plt.imshow(binary, cmap="gray")
            plt.title("Binary image - no contour found")
            plt.show()
        return None

    # Get the corners of the board
    corners, debug_img = get_board_corners(board_contour, image)

    if debug:
        plt.figure(figsize=(10, 10))
        plt.imshow(debug_img)
        plt.title("Detected Corners")
        plt.show()

    # Apply perspective transformation
    warped = perspective_transform(enhanced, corners)

    if debug:
        plt.figure(figsize=(8, 8))
        plt.imshow(warped, cmap="gray")
        plt.title("Warped Chessboard")
        plt.show()

    # Segment the chessboard
    squares = segment_chessboard(warped)

    if debug:
        # Show a sample of squares
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()

        for i in range(8):
            row, col = i // 4, i % 4
            square_img = squares[row][col]
            axes[i].imshow(square_img, cmap="gray")
            axes[i].set_title(f"Square {row},{col}")
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()

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
        "P": "♙",
        "N": "♘",
        "B": "♗",
        "R": "♖",
        "Q": "♕",
        "K": "♔",
        "p": "♟",
        "n": "♞",
        "b": "♝",
        "r": "♜",
        "q": "♛",
        "k": "♚",
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
        "P": "♙",
        "N": "♘",
        "B": "♗",
        "R": "♖",
        "Q": "♕",
        "K": "♔",
        "p": "♟",
        "n": "♞",
        "b": "♝",
        "r": "♜",
        "q": "♛",
        "k": "♚",
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


def process_board_with_hybrid_detection(pdf_path, debug=False):
    """
    Process a chess board image and detect pieces using hybrid approach.

    Args:
        pdf_path: Path to the PDF file
        debug: Whether to show debug visualizations

    Returns:
        FEN notation, board squares for visualization, and piece grid
    """
    # Process the board to get squares
    squares = process_chessboard(pdf_path, debug=debug)

    if squares is None:
        return None, None, None

    # Use hybrid detection to identify pieces
    piece_grid = detect_chess_pieces_hybrid(squares)

    # Convert to FEN notation
    fen = convert_board_to_fen(piece_grid)

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
        "P": (0, 0, 0),  # Black color for white pieces (uppercase)
        "N": (0, 0, 0),
        "B": (0, 0, 0),
        "R": (0, 0, 0),
        "Q": (0, 0, 0),
        "K": (0, 0, 0),
        "p": (255, 255, 255),  # White color for black pieces (lowercase)
        "n": (255, 255, 255),
        "b": (255, 255, 255),
        "r": (255, 255, 255),
        "q": (255, 255, 255),
        "k": (255, 255, 255),
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
            if (
                stats[i, cv2.CC_STAT_AREA] > 10
            ):  # Lower threshold to catch smaller marks
                valid_components.append(i)

        if not valid_components:
            return "empty"

        # Check for subscript 'w' to determine color
        has_subscript_w = False
        for i in valid_components:
            # Get component's center y-position relative to image height
            center_y = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2
            rel_y = center_y / cropped.shape[0]

            # Get component's dimensions
            comp_width = stats[i, cv2.CC_STAT_WIDTH]
            comp_height = stats[i, cv2.CC_STAT_HEIGHT]

            # If component is in bottom half and small, it might be a 'w'
            if (
                rel_y > 0.5
                and comp_width < cropped.shape[1] / 4
                and comp_height < cropped.shape[0] / 4
            ):
                has_subscript_w = True
                break

        # Find the main component (usually the largest one)
        if len(valid_components) > 0:
            main_component = max(
                valid_components, key=lambda i: stats[i, cv2.CC_STAT_AREA]
            )

            # Get shape metrics
            width = stats[main_component, cv2.CC_STAT_WIDTH]
            height = stats[main_component, cv2.CC_STAT_HEIGHT]
            area = stats[main_component, cv2.CC_STAT_AREA]
            aspect = width / height if height > 0 else 1.0

            # Get relative position
            top = stats[main_component, cv2.CC_STAT_TOP]
            left = stats[main_component, cv2.CC_STAT_LEFT]
            rel_x = (left + width / 2) / cropped.shape[1]  # Relative x center
            rel_y = (top + height / 2) / cropped.shape[0]  # Relative y center

            # By default, assume a pawn
            piece_type = "P"

            # Look at key characteristics to determine piece type
            if aspect > 1.2:  # Wider than tall
                if area > 0.2 * cropped.shape[0] * cropped.shape[1]:
                    piece_type = "K"  # Larger shapes are kings
                else:
                    piece_type = "R"  # Rooks tend to be wide
            elif aspect < 0.6:  # Taller than wide
                if area < 0.05 * cropped.shape[0] * cropped.shape[1]:
                    piece_type = "P"  # Pawns tend to be smaller
                else:
                    piece_type = "Q"  # Queens tend to be taller
            else:  # Moderate aspect ratio
                # Look at position and size
                if rel_y < 0.4:  # Higher in the square
                    piece_type = "N"  # Knights tend to be higher
                elif area > 0.1 * cropped.shape[0] * cropped.shape[1]:
                    piece_type = "B"  # Bishops tend to be larger
                else:
                    piece_type = "P"  # Default to pawn
        else:
            # No valid components (shouldn't happen, but just in case)
            piece_type = "P"

        # Apply color based on whether we found a subscript 'w'
        return piece_type if has_subscript_w else piece_type.lower()

    # Process each square in the grid
    for row in range(8):
        for col in range(8):
            square_img = squares[row][col]

            if has_text(square_img):
                piece_grid[row][col] = identify_piece(square_img)
            else:
                piece_grid[row][col] = "empty"

    return piece_grid


# End-to-End Pipeline - Mac Optimized


def main():
    """
    Main function to run the entire pipeline.
    """
    # Determine if we're running in a notebook or as a script
    running_in_notebook = "ipykernel" in sys.modules

    print("=" * 50)
    print("Chess Board Recognition - Mac Optimized")
    print("=" * 50)

    # Check if there are PDF files to process
    if not pdf_files:
        print("No PDF files found in the boards directory.")
        return

    # Process a single PDF file as a test
    example_pdf = os.path.join(input_dir, pdf_files[0])
    print(f"\nTesting board processing on: {pdf_files[0]}")

    # Process the board with hybrid detection
    fen, squares, piece_grid = process_board_with_hybrid_detection(
        example_pdf, debug=running_in_notebook
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
        else:
            # In notebook, display the visualization
            visualize_detected_board(squares, piece_grid)
    else:
        print("Failed to process the board.")

    # Ask if user wants to process all files
    if running_in_notebook or input("\nProcess all PDF files? (y/n): ").lower() == "y":
        print("\nProcessing all PDF files...")
        results = {}

        for i, pdf_file in enumerate(pdf_files):
            pdf_path = os.path.join(input_dir, pdf_file)
            print(f"Processing {i+1}/{len(pdf_files)}: {pdf_file}")

            try:
                fen, squares, piece_grid = process_board_with_hybrid_detection(pdf_path)

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
