import json
import numpy as np

from paths import OUTPUTS_ROOT


def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def calculate_metrics(ground_truth, predictions):
    # Initialize variables to store sum of errors for RMSE and relRMSE
    sum_sq_error = {}
    sum_rel_sq_error = {}
    non_zero_count = {}
    total_images = len(ground_truth)
    if total_images == 0:
        raise ValueError("ground_truth must contain at least one image")
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
    # The paper defines mRMSE as the mean of the per-category RMSE values.
    # The nonzero variants average only over categories with nonzero examples.
    m_rmse = float(np.mean(list(rmse.values()))) if rmse else 0.0
    m_rmse_nz = float(np.mean(list(rmse_nz.values()))) if rmse_nz else 0.0
    m_rel_rmse = float(np.mean(list(rel_rmse.values()))) if rel_rmse else 0.0
    m_rel_rmse_nz = (
        float(np.mean(list(rel_rmse_nz.values()))) if rel_rmse_nz else 0.0
    )

    return rmse, rmse_nz, rel_rmse, rel_rmse_nz, m_rmse, m_rmse_nz, m_rel_rmse, m_rel_rmse_nz



if __name__ == "__main__":
    ground_truth_file_path = OUTPUTS_ROOT / "animals" / "animal_gt.json"
    prediction_file_path = OUTPUTS_ROOT / "animals" / "animal_pred_1.json"

    ground_truth = load_json_file(ground_truth_file_path)
    predictions = load_json_file(prediction_file_path)
    metrics = calculate_metrics(ground_truth, predictions)
    print("Mean RMSE:", metrics[4])
    print("Mean RMSE-nz:", metrics[5])
    print("Mean relRMSE:", metrics[6])
    print("Mean relRMSE-nz:", metrics[7])
