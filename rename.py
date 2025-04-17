import os
import shutil
import tempfile

# Path to the folder containing the PDF files
folder_path = "/Users/emmetthintz/Documents/classes/ml/final/boards/"

# Create a temporary directory
temp_dir = tempfile.mkdtemp()
print(f"Created temporary directory: {temp_dir}")

try:
    # Get a list of all PDF files in the folder
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    pdf_files.sort()

    # First, copy all files to the temporary directory with their new names
    for index, filename in enumerate(pdf_files):
        old_path = os.path.join(folder_path, filename)
        temp_path = os.path.join(temp_dir, f"board{index + 1}.pdf")
        shutil.copy2(old_path, temp_path)
        print(f"Copied: {filename} -> temp/board{index + 1}.pdf")

    # Then move all the renamed files back to the original directory
    for index in range(1, len(pdf_files) + 1):
        temp_path = os.path.join(temp_dir, f"board{index}.pdf")
        new_path = os.path.join(folder_path, f"board{index}.pdf")
        # Remove the destination file if it already exists
        if os.path.exists(new_path):
            os.remove(new_path)
        shutil.move(temp_path, new_path)
        print(f"Moved: temp/board{index}.pdf -> board{index}.pdf")

    # Count files to verify
    final_count = len([f for f in os.listdir(folder_path) if f.endswith(".pdf")])
    print(f"Total files in {folder_path}: {final_count}")

finally:
    # Clean up the temporary directory
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary directory: {temp_dir}")
