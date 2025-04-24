import os

def delete_all_but_every_10th_image(folder_path):
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

    try:
        # Check if folder exists
        if not os.path.isdir(folder_path):
            print(f"Error: The folder '{folder_path}' does not exist.")
            return

        # List and filter image files
        image_files = sorted([
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in image_extensions
        ])

        if not image_files:
            print(f"No image files found in '{folder_path}'.")
            return

        # Keep every 10th image (0, 10, 20, ...)
        for idx, filename in enumerate(image_files):
            if idx % 10 != 0:  # Delete everything that is not a 10th image
                file_path = os.path.join(folder_path, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

        print("Cleanup complete.")

    except Exception as e:
        print(f"Unexpected error: {e}")

# Example usage — use raw string to handle Windows path
delete_all_but_every_10th_image(
    r'C:\Users\chris\Documents\learning\ncstate\underwater-robotics\SW8S-Rust\tests\vision\resources\ocean_cleanup_training\dataset\shark\val'
)
