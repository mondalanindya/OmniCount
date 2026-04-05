#Code for average size 

# from PIL import Image
# import os
# import argparse

# def calculate_avg_dimensions(folder_path):
#     total_width = 0
#     total_height = 0
#     image_count = 0
    
#     # Iterate over all files in the folder
#     for file_name in os.listdir(folder_path):
#         try:
#             # Construct full file path
#             file_path = os.path.join(folder_path, file_name)
#             # Open the image file
#             with Image.open(file_path) as img:
#                 # Increment total dimensions and count
#                 total_width += img.width
#                 total_height += img.height
#                 image_count += 1
#         except IOError:
#             # Skip files that are not images
#             print(f"Skipping file (not an image): {file_name}")
#             continue
    
#     # Calculate average dimensions
#     if image_count > 0:
#         avg_width = total_width / image_count
#         avg_height = total_height / image_count
#         return avg_width, avg_height
#     else:
#         return 0, 0

# def main():
#     parser = argparse.ArgumentParser(description='Calculate the average dimensions of images in a folder.')
#     parser.add_argument('folder_path', type=str, help='The path to the folder containing the images.')
    
#     args = parser.parse_args()
    
#     avg_width, avg_height = calculate_avg_dimensions(args.folder_path)
#     print(f"Average Width: {avg_width}, Average Height: {avg_height}")

# if __name__ == '__main__':
#     main()

#size code ends here 

#average count code starts here

import json
import argparse

def calculate_average_count(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    total_counts = 0
    total_images = len(data)
    
    for image_id, classes in data.items():
        for class_name, count in classes.items():
            total_counts += count
    
    if total_images > 0:
        average_count = total_counts / total_images
        return average_count
    else:
        return 0

def main():
    parser = argparse.ArgumentParser(description='Calculate the average count of classes per image in a JSON file.')
    parser.add_argument('json_file_path', type=str, help='The path to the JSON file.')
    
    args = parser.parse_args()
    
    average_count = calculate_average_count(args.json_file_path)
    print(f"Average Count per Image: {average_count}")

if __name__ == '__main__':
    main()
#average count code ends here

# import json
# import argparse

# def calculate_total_objects(json_file_path):
#     with open(json_file_path, 'r') as file:
#         data = json.load(file)
    
#     total_objects = 0
    
#     for image_id, classes in data.items():
#         for class_name, count in classes.items():
#             total_objects += count
    
#     return total_objects

# def main():
#     parser = argparse.ArgumentParser(description='Calculate the total number of objects across all images in a JSON file.')
#     parser.add_argument('json_file_path', type=str, help='The path to the JSON file.')
    
#     args = parser.parse_args()
    
#     total_objects = calculate_total_objects(args.json_file_path)
#     print(f"Total number of objects: {total_objects}")

# if __name__ == '__main__':
#     main()
