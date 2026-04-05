import os
import re

from paths import OUTPUTS_ROOT

directory = str(OUTPUTS_ROOT / "animals" / "images")

# Pattern to extract the number from the filename
pattern = r"\d+"

for filename in os.listdir(directory):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp', '.jfif', '.avif', '.tiff', '.bmp', '.svg', '.JPG', )):
        match = re.search(pattern, filename)
        if match:
            image_number = match.group()
            new_filename = f"{image_number}.jpg"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)

            os.rename(old_path, new_path)
            print(f"Renamed '{filename}' to '{new_filename}'")

print("Renaming completed.")
