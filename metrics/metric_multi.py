import json
import numpy as np


def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def calculate_metrics(ground_truth, predictions):
    # Initialize variables to store sum of errors for RMSE and relRMSE
    sum_sq_error = {}
    sum_rel_sq_error = {}
    non_zero_count = {}
    total_images = len(ground_truth)
    sum_final = 0
    # Initialize sums for non-zero counts
    for key in ground_truth.values():
        for class_name in key.keys():
            sum_sq_error[class_name] = 0
            sum_rel_sq_error[class_name] = 0
            non_zero_count[class_name] = 0

    # Calculate squared errors
    for image_id, counts in ground_truth.items():
        pred_counts = predictions.get(image_id, {})
        for class_name, gt_count in counts.items():
            pred_count = pred_counts.get(class_name, 0)
            diff = abs(pred_count - gt_count)
            # print(image_id, diff)
            sum_final+=diff
            sum_sq_error[class_name] += diff ** 2
            sum_rel_sq_error[class_name] += (diff ** 2) / (gt_count + 1)
            
            if gt_count > 0:
                non_zero_count[class_name] += 1
    print("Sum of absolute differences:", sum_final)
    # Calculate metrics
    rmse = {class_name: np.sqrt(sum_sq_error[class_name] / total_images) for class_name in sum_sq_error}
    rmse_nz = {class_name: np.sqrt(sum_sq_error[class_name] / non_zero_count[class_name]) for class_name in sum_sq_error if non_zero_count[class_name] > 0}
    rel_rmse = {class_name: np.sqrt(sum_rel_sq_error[class_name] / total_images) for class_name in sum_rel_sq_error}
    rel_rmse_nz = {class_name: np.sqrt(sum_rel_sq_error[class_name] / non_zero_count[class_name]) for class_name in sum_rel_sq_error if non_zero_count[class_name] > 0}

    # Calculate mean metrics
    m_rmse = 0
    m_rmse_nz =0
    m_rel_rmse = 0
    m_rel_rmse_nz = 0

    for i in range(len(list(rmse.values()))):
        m_rmse+=list(rmse.values())[i]
    
    for j in range(len([value for key, value in rmse_nz.items()])):
        m_rmse_nz+= [value for key, value in rmse_nz.items()][j]

    for k in range(len(list(rel_rmse.values()))):
        m_rel_rmse+=list(rel_rmse.values())[k]

    for l in range(len([value for key, value in rel_rmse_nz.items()])):
        m_rel_rmse_nz+= [value for key, value in rel_rmse_nz.items()][l]

    m_rmse = m_rmse/total_images
    m_rmse_nz = m_rmse_nz/total_images
    m_rel_rmse = m_rel_rmse/total_images
    m_rel_rmse_nz = m_rel_rmse_nz/total_images


    # m_rmse = list(rmse.values())/total_images
    # m_rmse_nz = [value for key, value in rmse_nz.items()]/total_images
    # m_rel_rmse = list(rel_rmse.values())/total_images
    # m_rel_rmse_nz = [value for key, value in rel_rmse_nz.items()] / total_images

    return rmse, rmse_nz, rel_rmse, rel_rmse_nz, m_rmse, m_rmse_nz, m_rel_rmse, m_rel_rmse_nz



ground_truth_file_path = "/home/amondal/Codes/omnicount/outputs_sota/animals/animal_gt.json"
# prediction_file_path = "/home/amondal/Codes/voting/outputs_sota/FSC_test_split_multisingle/fsc_data_pred_multisingle.json"
# prediction_file_path = "/home/amondal/Codes/voting/baselines/tfoc_original/fsc_data_multiple_tfoc.json"
prediction_file_path = "/home/amondal/Codes/omnicount/outputs_sota/animals/animal_pred_1.json"

# Correctly load the JSON files using the load_json_file function
ground_truth = load_json_file(ground_truth_file_path)
predictions = load_json_file(prediction_file_path)

# Now, the ground_truth and predictions are correctly loaded dictionaries that can be passed to calculate_metrics
metrics = calculate_metrics(ground_truth, predictions)
# print("RMSE:", metrics[0])
# print("RMSE-nz:", metrics[1])
# print("relRMSE:", metrics[2])
# print("relRMSE-nz:", metrics[3])
print("Mean RMSE:", metrics[4])
print("Mean RMSE-nz:", metrics[5])
print("Mean relRMSE:", metrics[6])
print("Mean relRMSE-nz:", metrics[7])


