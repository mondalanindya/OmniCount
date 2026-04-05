import os
import cv2
import numpy as np
from PIL import Image
import random
from scipy import ndimage
from paths import OUTPUTS_ROOT

base_rgb_path = str(OUTPUTS_ROOT / "animals" / "images")
base_mask_path = str(OUTPUTS_ROOT / "animals" / "binary_masks")
base_depth_path = str(OUTPUTS_ROOT / "animals" / "depth" / "depth_bw")
output_masks_path = str(OUTPUTS_ROOT / "animals" / "refined_bin_masks")  # New path for saving binary masks

# Ensure output directory exists
os.makedirs(output_masks_path, exist_ok=True)

# Function to process each binary mask
def process_binary_mask(binary_mask, other_masks, depth_map_array_normalized, image_id):
    is_valid_sem = 1
    edge_indices = np.where(binary_mask == [255])
    if len(edge_indices[0]) == 0:
        is_valid_sem = 0
        print('No edge indices found for image:', image_id)

    edge_indices = zip(edge_indices[0], edge_indices[1])

    mask_indices = np.nonzero(binary_mask)
    mean_depth = np.mean(depth_map_array_normalized[mask_indices])
    refined_mask = np.zeros_like(binary_mask)
    tolerance = 0.2
    win = 5
    if is_valid_sem == 1:
        for x, y in edge_indices:
            for k in range(x-win, x+win):
                for j in range(y-win, y+win):
                    if 0 <= k < binary_mask.shape[0] and 0 <= j < binary_mask.shape[1]:
                        if any(other_mask[k, j] == 255 for other_mask in other_masks):
                            continue
                        if abs(depth_map_array_normalized[k, j] - mean_depth) < tolerance:
                            refined_mask[k, j] = 255
    else:
        valid_depth_idx = np.where(depth_map_array_normalized < 0.5)
        valid_depth_idx = zip(valid_depth_idx[0], valid_depth_idx[1])
        for x, y in valid_depth_idx:
            refined_mask[x, y] = 255

    # Apply morphological operations
    radius = 5
    strel = ndimage.generate_binary_structure(2, 1)
    strel = ndimage.iterate_structure(strel, radius)
    bw_opened = ndimage.binary_opening(refined_mask, structure=strel)

    refined_mask = bw_opened.astype(np.uint8) * 255

    return refined_mask

# Function to save binary mask
def save_binary_mask(mask, class_name, image_id):
    mask_output_dir = os.path.join(output_masks_path, image_id)
    os.makedirs(mask_output_dir, exist_ok=True)
    mask_output_path = os.path.join(mask_output_dir, f'mask_{class_name}.png')
    cv2.imwrite(mask_output_path, mask)

# Process each image ID
for image_id in os.listdir(base_mask_path):
    print('Now processing image:', image_id)
    depth_img_path = os.path.join(base_depth_path, f'{image_id}_pred.png')

    if not os.path.exists(depth_img_path):
        continue

    # Load depth map
    depth_map_array = np.array(Image.open(depth_img_path))
    depth_map_array_normalized = (depth_map_array - depth_map_array.min()) / (depth_map_array.max() - depth_map_array.min())

    # Load and process each class mask
    class_mask_paths = os.listdir(os.path.join(base_mask_path, image_id))
    for class_mask in class_mask_paths:
        binary_mask_path = os.path.join(base_mask_path, image_id, class_mask)
        binary_mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)

        _, binary_mask_otsu = cv2.threshold(binary_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        other_masks = [cv2.imread(os.path.join(base_mask_path, image_id, other_mask), cv2.IMREAD_GRAYSCALE) for other_mask in class_mask_paths if other_mask != class_mask]

        # Process and save each mask
        refined_mask = process_binary_mask(binary_mask_otsu, other_masks, depth_map_array_normalized, image_id)
        class_name = class_mask.split('_')[-1].split('.')[0]  # Extract class name
        save_binary_mask(refined_mask, class_name, image_id)
