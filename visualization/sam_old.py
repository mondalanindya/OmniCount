# import os
# import cv2
# import numpy as np
# import torch
# from PIL import Image
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# # SAM Model Setup
# sam_checkpoint = "sam_vit_h_4b8939.pth"
# model_type = "vit_h"
# device = "cuda"
# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
# mask_generator = SamAutomaticMaskGenerator(
#     model=sam,
#     points_per_side=32,
#     pred_iou_thresh=0.88,
#     stability_score_thresh=0.85,
#     stability_score_offset=1.0,
#     box_nms_thresh=0.5,
#     crop_n_layers=0,
#     crop_nms_thresh=0.7,
#     crop_overlap_ratio= 512 / 1500,
#     crop_n_points_downscale_factor=1,
#     min_mask_region_area=25,
# )

# # Ensure output directory exists
# os.makedirs(output_sam_patches_path, exist_ok=True)

# def overlay_and_save(image, mask, output_path, filename):
#     overlayed_image = np.where(mask[..., None], image, np.zeros_like(image))
#     cv2.imwrite(os.path.join(output_path, filename), cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR))

# # Process each patch
# for root, dirs, files in os.walk(base_patches_path):
#     for file in files:
#         patch_path = os.path.join(root, file)
#         image = cv2.imread(patch_path)
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Check for minimum size requirements or skip
#         if image_rgb.shape[0] < 32 or image_rgb.shape[1] < 32:
#             print(f"Skipping {patch_path}: Image too small for SAM model.")
#             continue

#         try:
#             # Generate masks with SAM
#             masks = mask_generator.generate(image_rgb)
#         except Exception as e:
#             print(f"Error processing {patch_path}: {e}")
#             continue

#         # Extract the relative path and create corresponding output directory
#         relative_path = os.path.relpath(root, base_patches_path)
#         output_path = os.path.join(output_sam_patches_path, relative_path)
#         os.makedirs(output_path, exist_ok=True)

#         # Overlay masks and save
#         for i, mask_dict in enumerate(masks):
#             mask = mask_dict.get('segmentation', None)
#             if mask is None:
#                 continue

#             if len(mask.shape) > 2:
#                 mask = mask[:, :, 0]

#             overlay_and_save(image_rgb, mask, output_path, f'sam_patch_{i}.png')

#################################################################################################################

import os
import cv2
import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from paths import OUTPUTS_ROOT

# # Define paths

# # Ensure output directory exists
# os.makedirs(output_sam_patches_path, exist_ok=True)

# # Load the SAM model
# sam_checkpoint = "sam_vit_h_4b8939.pth"
# model_type = "vit_h"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
# mask_generator = SamAutomaticMaskGenerator(
#     model=sam,
#     points_per_side=32,
#     pred_iou_thresh=0.86,
#     stability_score_thresh=0.92,
#     crop_n_layers=1,
#     crop_n_points_downscale_factor=2,
#     min_mask_region_area=100,  # Requires open-cv to run post-processing
# )

# def overlay_and_save(image, mask, output_path, filename):
#     overlayed_image = np.where(mask[..., None], image, np.zeros_like(image))
#     cv2.imwrite(os.path.join(output_path, filename), cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR))

# # Iterate over each patch and generate SAM masks
# for root, dirs, files in os.walk(base_patches_path):
#     for file in files:
#         patch_path = os.path.join(root, file)
#         image = cv2.imread(patch_path)
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Generate masks with SAM
#         masks = mask_generator.generate(image_rgb)

#         # Create output directory
#         relative_path = os.path.relpath(root, base_patches_path)
#         output_path = os.path.join(output_sam_patches_path, relative_path)
#         os.makedirs(output_path, exist_ok=True)

#         # Overlay masks and save
#         for i, mask_dict in enumerate(masks):
#             mask = mask_dict.get('segmentation', None)
#             if mask is None:
#                 continue

#             if len(mask.shape) > 2:
#                 mask = mask[:, :, 0]

#             filename = f'{os.path.splitext(file)[0]}_{i}.png'
#             overlay_and_save(image_rgb, mask, output_path, filename)

import os
import cv2
import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Define paths
base_patches_path = str(OUTPUTS_ROOT / 'CARPK_test_split' / 'patches')
output_sam_patches_path = str(OUTPUTS_ROOT / 'CARPK_test_split' / 'sam_patches')

# Ensure output directory exists
os.makedirs(output_sam_patches_path, exist_ok=True)

# Load the SAM model
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side= 32,
    points_per_batch= 64,
    pred_iou_thresh=0.77,
    stability_score_thresh=0.85,
    stability_score_offset=1.0,
    box_nms_thresh=0.5,
    crop_n_layers=0,
    crop_nms_thresh=0.7,
    crop_overlap_ratio=512 / 1500,
    crop_n_points_downscale_factor=1,
    point_grids= None,
    min_mask_region_area= 25,
    output_mode="binary_mask",
)

def overlay_and_save(image, mask, output_path, filename):
    overlayed_image = np.where(mask[..., None], image, np.zeros_like(image))
    cv2.imwrite(os.path.join(output_path, filename), cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR))

# Iterate over each patch and generate SAM masks
for image_id in os.listdir(base_patches_path):
    image_id_path = os.path.join(base_patches_path, image_id)
    if os.path.isdir(image_id_path):
        for file in os.listdir(image_id_path):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                patch_path = os.path.join(image_id_path, file)
                image = cv2.imread(patch_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Extract class name from file name (e.g., 'sea shells' from 'patch_sea shells.png')
                class_name = file.split('_')[1].rsplit('.', 1)[0]

                # Generate masks with SAM
                masks = mask_generator.generate(image_rgb)

                # Create output directory for this class
                output_class_path = os.path.join(output_sam_patches_path, image_id, class_name)
                os.makedirs(output_class_path, exist_ok=True)

                # Overlay masks and save
                for i, mask_dict in enumerate(masks):
                    mask = mask_dict.get('segmentation', None)
                    if mask is None:
                        continue

                    if len(mask.shape) > 2:
                        mask = mask[:, :, 0]

                    filename = f'{class_name}_{i}.png'
                    overlay_and_save(image_rgb, mask, output_class_path, filename)

