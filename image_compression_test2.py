from PIL import Image
import os
from tqdm import tqdm

# Set the source folders
source_folders = ["data/vehicle_classifier_3Output/test"]

# Set the desired maximum resolution (width, height) in pixels
max_resolution = (224, 224)

for source_folder in source_folders:
    # Create the destination folder with the same name as the source folder
    dest_folder = os.path.join("data/vehicle_classifier_3Output/test/compressed")

    # Create the destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Get a list of all JPG files in the source folder
    image_files = [
        f
        for f in os.listdir(source_folder)
        if f.endswith(".jpg", ".jpeg", ".JPG", ".JPEG")
    ]

    # Create a progress bar
    progress_bar = tqdm(
        total=len(image_files),
        unit="image",
        unit_scale=True,
        unit_divisor=1024,
        position=0,
        leave=True,
    )

    # Loop through all JPG files in the source folder
    for filename in image_files:
        # Open the image
        image_path = os.path.join(source_folder, filename)
        image = Image.open(image_path)

        # Resize the image while maintaining aspect ratio
        if image.size > max_resolution:
            image.thumbnail(max_resolution, Image.LANCZOS)

        # Get the exif data from the original image
        exif_data = image.info.get("exif")

        # Compress the image
        compressed_image = image.copy()
        compressed_image.save(
            os.path.join(dest_folder, filename),
            optimize=True,
            quality=85,
            exif=exif_data,
        )

        # Update the progress bar
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    print(f"Image compression completed for {source_folder}.")
