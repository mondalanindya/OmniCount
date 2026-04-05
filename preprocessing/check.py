# import os

import os

from paths import OUTPUTS_ROOT

def find_missing_images(depth_bw_dir, binary_masks_dir):
    # Get all files in the depth_bw directory
    depth_bw_files = os.listdir(depth_bw_dir)
    
    # Get all directories in the binary_masks directory
    binary_masks_dirs = os.listdir(binary_masks_dir)
    
    # Extract numeric identifiers from the depth_bw filenames
    depth_bw_ids = set(int(file.split('_')[0]) for file in depth_bw_files if file.endswith('.png'))
    
    # Extract numeric identifiers from the binary_masks directory names
    binary_masks_ids = set(int(dir_name) for dir_name in binary_masks_dirs)
    
    # Find ids present in depth_bw but not in binary_masks
    missing_ids = depth_bw_ids - binary_masks_ids
    
    return missing_ids

# Specify the paths to your directories
depth_bw_dir = OUTPUTS_ROOT / 'FSC_test_split_multi' / 'depth' / 'depth_bw'
binary_masks_dir = OUTPUTS_ROOT / 'FSC_test_split_multi' / 'binary_masks'

# Call the function and print the result
missing_ids = find_missing_images(depth_bw_dir, binary_masks_dir)
print("Image IDs in 'depth_bw' but not in 'binary_masks':", missing_ids)
