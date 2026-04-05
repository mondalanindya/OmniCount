# import os
# import cv2
# import numpy as np
# from PIL import Image
# import random

# # Ensure output directory exists
# os.makedirs(output_patches_path, exist_ok=True)

# # Function to process each binary mask
# def process_binary_mask(binary_mask, other_masks, depth_map_array_normalized):
#     edge_indices = np.where(binary_mask == [255])
#     edge_indices = zip(edge_indices[0], edge_indices[1])
#     mask_indices = np.nonzero(binary_mask)
#     mean_depth = np.mean(depth_map_array_normalized[mask_indices])
#     refined_mask = np.zeros_like(binary_mask)
#     tolerance = 0.1
#     for x, y in edge_indices:
#         for k in range(x-5, x+5):
#             for j in range(y-5, y+5):
#                 if 0 <= k < binary_mask.shape[0] and 0 <= j < binary_mask.shape[1]:
#                     if any(other_mask[k, j] == 255 for other_mask in other_masks):
#                         continue
#                     if abs(depth_map_array_normalized[k, j] - mean_depth) < tolerance:
#                         refined_mask[k, j] = 255
#     return refined_mask


# # Function to overlay mask on image and extract patch
# def extract_patch(rgb_image, mask, class_name, image_id):
#     # Convert mask to boolean array
#     boolean_mask = mask.astype(bool)

#     # Extract the area of the image where the mask is true
#     patch = rgb_image.copy()
#     patch[~boolean_mask] = 0  # Set the area outside the mask to black

#     # Save the patch
#     patch_output_path = os.path.join(output_patches_path, image_id, f'patch_{class_name}.png')
#     cv2.imwrite(patch_output_path, patch)

# # Process each image ID
# for image_id in os.listdir(base_mask_path):
#     rgb_image_path = os.path.join(base_rgb_path, f'{image_id}.png')
#     depth_img_path = os.path.join(base_depth_path, f'{image_id}_pred.png')

#     if not os.path.exists(rgb_image_path) or not os.path.exists(depth_img_path) or not os.path.exists(os.path.join(output_patches_path, image_id)):
#         continue

#     # Load RGB image and depth map
#     rgb_image = cv2.imread(rgb_image_path)
#     depth_map_array = np.array(Image.open(depth_img_path))
#     depth_map_array_normalized = (depth_map_array - depth_map_array.min()) / (depth_map_array.max() - depth_map_array.min())

#     # Resize depth map to match RGB image dimensions if necessary
#     if depth_map_array_normalized.shape[:2] != rgb_image.shape[:2]:
#         depth_map_array_normalized = cv2.resize(depth_map_array_normalized, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)

#     # Create output directory for patches
#     image_patches_path = os.path.join(output_patches_path, image_id)
#     os.makedirs(image_patches_path, exist_ok=True)

#     # Load and process each class mask
#     class_mask_paths = os.listdir(os.path.join(base_mask_path, image_id))
#     for class_mask in class_mask_paths:
#         binary_mask_path = os.path.join(base_mask_path, image_id, class_mask)
#         binary_mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)

#         # Resize binary mask to match RGB image dimensions if necessary
#         if binary_mask.shape[:2] != rgb_image.shape[:2]:
#             binary_mask = cv2.resize(binary_mask, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)

#         _, binary_mask_otsu = cv2.threshold(binary_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         other_masks = [cv2.imread(os.path.join(base_mask_path, image_id, other_mask), cv2.IMREAD_GRAYSCALE) for other_mask in class_mask_paths if other_mask != class_mask]

#         # Process and overlay each mask
#         refined_mask = process_binary_mask(binary_mask_otsu, other_masks, depth_map_array_normalized)
#         class_name = class_mask.split('_')[-1].split('.')[0]  # Extract class name
#         extract_patch(rgb_image, refined_mask, class_name, image_id)


import os
import cv2
import numpy as np
from PIL import Image
import random
from paths import OUTPUTS_ROOT

base_rgb_path = str(OUTPUTS_ROOT / "animals" / "images")
base_mask_path = str(OUTPUTS_ROOT / "animals" / "binary_masks")
base_depth_path = str(OUTPUTS_ROOT / "animals" / "depth" / "depth_bw")
output_patches_path = str(OUTPUTS_ROOT / "animals" / "patches")
# Ensure output directory exists
os.makedirs(output_patches_path, exist_ok=True)

# Function to process each binary mask
def process_binary_mask(binary_mask, other_masks, depth_map_array_normalized, image_id):
    is_valid_sem = 1
    edge_indices = np.where(binary_mask == [255])
    # print('edge_indices:', edge_indices)
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
                        # print('k:', k, 'j:', j)
                        if any(other_mask[k, j] == 255 for other_mask in other_masks):
                            continue
                        if abs(depth_map_array_normalized[k, j] - mean_depth) < tolerance:
                            refined_mask[k, j] = 255
    else:
        valid_depth_idx = np.where(depth_map_array_normalized < 0.5)
        valid_depth_idx = zip(valid_depth_idx[0], valid_depth_idx[1])
        for x, y in valid_depth_idx:
            refined_mask[x, y] = 255

    from scipy import ndimage
    radius = 5
    def create_disk(radius: int):
        """
        Create a circular structuring element.
        This is essentially the binary/morphological equivalent the filter kernel.
        """
        r = radius
        x = np.arange(-r, r+1)
        y = np.arange(-r, r+1)
        y, x = np.meshgrid(x, x)
        return x*x + y*y <= r*r

    # Create structuring element
    strel = create_disk(radius)

    # Load image & process
    bw = refined_mask
    bw_opened = ndimage.binary_opening(bw, structure=strel)
    # bw_closed = ndimage.binary_closing(bw, structure=strel)

    refined_mask = bw_opened

    return refined_mask


# Function to overlay mask on image and extract patch
def extract_patch(rgb_image, mask, class_name, image_id):
    # Convert mask to boolean array
    boolean_mask = mask.astype(bool)

    # Extract the area of the image where the mask is true
    patch = rgb_image.copy()
    patch[~boolean_mask] = 0  # Set the area outside the mask to black

    # Save the patch
    patch_output_path = os.path.join(output_patches_path, image_id, f'patch_{class_name}.png')
    cv2.imwrite(patch_output_path, patch)

# Process each image ID
for image_id in os.listdir(base_mask_path):
    # if image_id == '4966':
    rgb_image_path = os.path.join(base_rgb_path, f'{image_id}.jpg')
    depth_img_path = os.path.join(base_depth_path, f'{image_id}_pred.png')
    
    if not os.path.exists(depth_img_path):
        print(f"Depth image for {image_id} does not exist, skipping.")
        continue

    if os.path.exists(os.path.join(output_patches_path, image_id)):
        continue

    print('Now processing image:', image_id)
    # if not os.path.exists(rgb_image_path) or not os.path.exists(depth_img_path) or not os.path.exists(os.path.join(output_patches_path, image_id)):
    #     continue

    # Load RGB image and depth map
    rgb_image = cv2.imread(rgb_image_path)
    if rgb_image is None:
        print(f"RGB image for {image_id} could not be loaded, skipping.")
        continue
    depth_map_array = np.array(Image.open(depth_img_path))
    depth_map_array_normalized = (depth_map_array - depth_map_array.min()) / (depth_map_array.max() - depth_map_array.min())

    # Resize depth map to match RGB image dimensions if necessary
    if depth_map_array_normalized.shape[:2] != rgb_image.shape[:2]:
        depth_map_array_normalized = cv2.resize(depth_map_array_normalized, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create output directory for patches
    image_patches_path = os.path.join(output_patches_path, image_id)
    os.makedirs(image_patches_path, exist_ok=True)

    # Load and process each class mask
    class_mask_paths = os.listdir(os.path.join(base_mask_path, image_id))
    for class_mask in class_mask_paths:
        binary_mask_path = os.path.join(base_mask_path, image_id, class_mask)
        binary_mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize binary mask to match RGB image dimensions if necessary
        if binary_mask.shape[:2] != rgb_image.shape[:2]:
            binary_mask = cv2.resize(binary_mask, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        _, binary_mask_otsu = cv2.threshold(binary_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        other_masks = [cv2.imread(os.path.join(base_mask_path, image_id, other_mask), cv2.IMREAD_GRAYSCALE) for other_mask in class_mask_paths if other_mask != class_mask]

        # Process and overlay each mask
        refined_mask = process_binary_mask(binary_mask_otsu, other_masks, depth_map_array_normalized, image_id)
        class_name = class_mask.split('_')[-1].split('.')[0]  # Extract class name
        extract_patch(rgb_image, refined_mask, class_name, image_id)
