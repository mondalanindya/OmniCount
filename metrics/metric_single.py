# import json
# import numpy as np

# # Load the ground truth and predictions from JSON files
# with open('ground_truth.json', 'r') as f:
#     ground_truth = json.load(f)

# with open('predictions.json', 'r') as f:
#     predictions = json.load(f)

# # Initialize lists to store the errors
# absolute_errors = []
# squared_errors = []

# # Iterate over the ground truth items
# for image_id, objects in ground_truth.items():
#     for object_type, count in objects.items():
#         # Get the corresponding prediction count, defaulting to 0 if not found
#         predicted_count = predictions.get(image_id, {}).get(object_type, 0)
        
#         # Calculate the absolute and squared errors
#         error = abs(count - predicted_count)
#         squared_error = (count - predicted_count) ** 2
        
#         # Append the errors to the lists
#         absolute_errors.append(error)
#         squared_errors.append(squared_error)

# # Calculate MAE and RMSE
# mae = np.mean(absolute_errors)
# rmse = np.sqrt(np.mean(squared_errors))

# print(f"MAE: {mae}")
# print(f"RMSE: {rmse}")


import json
import numpy as np
import math

# Load the ground truth and predictions from JSON files
with open('/home/amondal/Codes/omnicount/outputs_sota/FSC_test_split_single/fsc_data_single_gt.json', 'r') as f:
    ground_truth = json.load(f)

with open('/home/amondal/Codes/vqa/fsc_single/gt/vilt.json', 'r') as f:
    predictions = json.load(f)

# Initialize variables to store the errors and counts
absolute_errors = 0
squared_errors = 0
total_actual_counts = 0  # For NAE
sum_relative_errors = 0  # For SRE
obj_count = 0

# Iterate over the ground truth items
for image_id, objects in ground_truth.items():
    if image_id in predictions:
        for object_type, count in objects.items():
            predicted_count = predictions[image_id].get(object_type, 0)
            obj_count += 1
            
            # Calculate the absolute and squared errors
            error = abs(count - predicted_count)
            squared_error = (count - predicted_count) ** 2
            
            # Update the total errors and counts
            absolute_errors += error
            squared_errors += squared_error
            total_actual_counts += count  # For NAE
            
            # Avoid division by zero for SRE calculation
            if count > 0:
                sum_relative_errors += error / count

# Calculate MAE and RMSE
mae = absolute_errors / obj_count
rmse = math.sqrt(squared_errors / obj_count)

# Calculate NAE and SRE
nae = absolute_errors / total_actual_counts
sre = sum_relative_errors / obj_count

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"NAE: {nae}")
print(f"SRE: {sre}")

