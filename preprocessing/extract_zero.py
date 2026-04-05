import os
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from paths import OUTPUTS_ROOT

base_rgb_path = str(OUTPUTS_ROOT / "animals" / "images")
base_mask_path = str(OUTPUTS_ROOT / "animals" / "binary_masks")
base_depth_path = str(OUTPUTS_ROOT / "animals" / "depth" / "depth_bw")
output_patches_path = str(OUTPUTS_ROOT / "animals" / "patches")

# Ensure output directories exist
os.makedirs(output_patches_path, exist_ok=True)

def create_disk(radius):
    """Create a circular structuring element."""
    x, y = np.ogrid[-radius: radius+1, -radius: radius+1]
    return x**2 + y**2 <= radius**2

def process_and_save_refined_mask(binary_mask, other_masks, depth_map_array_normalized, image_id, class_mask):
    """Process each binary mask and overwrite the original if it's entirely black."""
    # Initial check for completely black mask
    if np.all(binary_mask == 0):
        print(f'Processing completely black mask for image: {image_id}, mask: {class_mask}')

        mean_depth = np.mean(depth_map_array_normalized)
        print(mean_depth)
        refined_mask = np.zeros_like(binary_mask)
        tolerance = 0.2

        # Apply depth-based refinement
        for x in range(binary_mask.shape[0]):
            for y in range(binary_mask.shape[1]):
                if abs(depth_map_array_normalized[x, y] - mean_depth) < tolerance:
                    refined_mask[x, y] = 255

        # Apply morphological opening to refine mask
        strel = create_disk(5)
        refined_mask = ndimage.binary_opening(refined_mask, structure=strel).astype(np.uint8) * 255
        depth_map = depth_map_array_normalized.astype(np.uint8) * 255
        
        # Initialize recovered_mask to ensure it has a value
        recovered_mask = np.zeros_like(binary_mask)
        
        if len(other_masks) == 1:
            recovered_mask = depth_map - other_masks[0]
        elif len(other_masks) > 1:
            for other_mask in other_masks:
                if not np.all(other_mask == 0):
                    recovered_mask = depth_map - other_mask
                    break  # Ensure we exit the loop once a valid mask is subtracted

        refined_mask = recovered_mask
        # Replace the original binary mask with the refined mask
        binary_mask_output_path = os.path.join(base_mask_path, image_id, class_mask)
        cv2.imwrite(binary_mask_output_path, refined_mask)
        print(f'Replaced original mask with refined mask for {image_id}/{class_mask}')


# # Process each image ID
# for image_id in os.listdir(base_mask_path):
#     rgb_image_path = os.path.join(base_rgb_path, f'{image_id}.jpg')
#     depth_img_path = os.path.join(base_depth_path, f'{image_id}_pred.png')

#     # if not os.path.exists(rgb_image_path) or not os.path.exists(depth_img_path):
#     #     continue

#     # Load and normalize depth map
#     depth_map_array = np.array(Image.open(depth_img_path))
#     depth_map_array_normalized = (depth_map_array - np.min(depth_map_array)) / (np.max(depth_map_array) - np.min(depth_map_array))
#     # print(depth_map_array_normalized)

#     # Process each class mask
#     for class_mask in os.listdir(os.path.join(base_mask_path, image_id)):
#         binary_mask_path = os.path.join(base_mask_path, image_id, class_mask)
#         binary_mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)

#         # Skip if binary mask is not entirely black
#         if not np.all(binary_mask == 0):
#             continue

#         # Collect other masks for comparison
#         other_masks = [cv2.imread(os.path.join(base_mask_path, image_id, m), cv2.IMREAD_GRAYSCALE) for m in os.listdir(os.path.join(base_mask_path, image_id)) if m != class_mask]

#         # Process and save refined mask
#         process_and_save_refined_mask(binary_mask, other_masks, depth_map_array_normalized, image_id, class_mask)
        

# Process each image ID
for image_id in os.listdir(base_mask_path):
    rgb_image_path = os.path.join(base_rgb_path, f'{image_id}.jpg')
    depth_img_path = os.path.join(base_depth_path, f'{image_id}_pred.png')

    # Check if the depth image exists, skip processing this image ID if it doesn't
    if not os.path.exists(depth_img_path):
        print(f"Depth image for {image_id} does not exist, skipping.")
        continue

    # Load and normalize depth map
    depth_map_array = np.array(Image.open(depth_img_path))
    depth_map_array_normalized = (depth_map_array - np.min(depth_map_array)) / (np.max(depth_map_array) - np.min(depth_map_array))
    # print(depth_map_array_normalized)

    # Process each class mask
    for class_mask in os.listdir(os.path.join(base_mask_path, image_id)):
        binary_mask_path = os.path.join(base_mask_path, image_id, class_mask)
        binary_mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)

        # Skip if binary mask is not entirely black
        if not np.all(binary_mask == 0):
            continue

        # Collect other masks for comparison
        other_masks = [cv2.imread(os.path.join(base_mask_path, image_id, m), cv2.IMREAD_GRAYSCALE) for m in os.listdir(os.path.join(base_mask_path, image_id)) if m != class_mask]

        # Process and save refined mask
        process_and_save_refined_mask(binary_mask, other_masks, depth_map_array_normalized, image_id, class_mask)

