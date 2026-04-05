import cv2
import numpy as np
import os

from paths import OUTPUTS_ROOT

def generate_unique_colors(n):
    # Generate n unique colors
    np.random.seed(42)  # Ensure reproducible colors
    colors = np.random.randint(0, 255, (n, 3), dtype=np.uint8)
    return colors

def overlay_masks_dynamic_colors(image_path, masks_dir):
    original_image = cv2.imread(image_path)
    
    # List all mask files
    mask_files = []
    for subdir, dirs, files in os.walk(masks_dir):
        for file in files:
            if file.endswith(('.jpg', '.png')):
                mask_files.append(os.path.join(subdir, file))
    
    # Generate unique colors for each mask
    n_masks = len(mask_files)
    colors = generate_unique_colors(n_masks)

    for i, mask_path in enumerate(mask_files):
        # Load the binary mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Ensure the mask is loaded correctly
        if mask is None:
            print(f"Failed to load mask: {mask_path}")
            continue
        
        # Verify that mask is binary
        if not np.isin(np.unique(mask), [0, 255]).all():
            print(f"Non-binary mask detected: {mask_path}, correcting...")
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Create a colored mask
        colored_mask = np.zeros_like(original_image)
        colored_mask[mask == 255] = colors[i]
        
        # Overlay the colored mask on the original image with adjusted alpha for better visibility
        original_image = cv2.addWeighted(original_image, 1, colored_mask, 0.7, 0)  # Adjusted alpha for visibility

    return original_image

image_path = str(OUTPUTS_ROOT / 'animals' / 'images' / '5.jpg')
masks_dir = str(OUTPUTS_ROOT / 'animals' / 'sam_patches' / '5')
result_image_path = str(OUTPUTS_ROOT / 'animals' / 'vizs' / '5_mask.jpg')

resulting_image = overlay_masks_dynamic_colors(image_path, masks_dir)
cv2.imwrite(result_image_path, resulting_image)
