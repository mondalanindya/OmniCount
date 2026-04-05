# import os

# # Paths to the directories
# masks_path = '/home/amondal/Codes/omnicount/outputs_sota/FSC_test_split_multi/binary_masks'
# ref_masks_path = '/home/amondal/Codes/omnicount/outputs_sota/FSC_test_split_multi_old/sam_patches'

# # Retrieve the list of folder IDs in each directory
# masks_ids = next(os.walk(masks_path))[1]
# ref_masks_ids = next(os.walk(ref_masks_path))[1]

# # Convert lists to sets for easier comparison
# masks_ids_set = set(masks_ids)
# ref_masks_ids_set = set(ref_masks_ids)

# # Find IDs that are in masks but not in ref_masks and vice versa
# masks_not_in_ref = masks_ids_set - ref_masks_ids_set
# ref_not_in_masks = ref_masks_ids_set - masks_ids_set

# # Print the mismatched IDs
# if masks_not_in_ref:
#     print(f"IDs in masks but not in ref_masks: {masks_not_in_ref}")
# else:
#     print("All IDs in masks are present in ref_masks.")

# if ref_not_in_masks:
#     print(f"IDs in ref_masks but not in masks: {ref_not_in_masks}")
# else:
#     print("All IDs in ref_masks are present in masks.")


import os

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
depth_bw_dir = '/home/amondal/Codes/omnicount/outputs_sota/FSC_test_split_multi/depth/depth_bw'
binary_masks_dir = '/home/amondal/Codes/omnicount/outputs_sota/FSC_test_split_multi/binary_masks'

# Call the function and print the result
missing_ids = find_missing_images(depth_bw_dir, binary_masks_dir)
print("Image IDs in 'depth_bw' but not in 'binary_masks':", missing_ids)
