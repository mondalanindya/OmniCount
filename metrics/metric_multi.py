import json
import numpy as np

from paths import OUTPUTS_ROOT


def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def calculate_metrics(ground_truth, predictions):
    """Calculate category-averaged RMSE metrics for multi-label counts.

    Each category is evaluated over every image. Missing ground-truth or
    prediction entries are therefore treated as zero, so false positives and
    false negatives are both included.
    """
    total_images = len(ground_truth)
    if total_images == 0:
        raise ValueError("ground_truth must contain at least one image")

    class_names = {
        class_name
        for counts in ground_truth.values()
        for class_name in counts
    }
    class_names.update(
        class_name
        for counts in predictions.values()
        for class_name in counts
    )
    class_names = sorted(class_names)
    sum_sq_error = {class_name: 0.0 for class_name in class_names}
    sum_rel_sq_error = {class_name: 0.0 for class_name in class_names}
    non_zero_count = {class_name: 0 for class_name in class_names}
    sum_final = 0

    for image_id, counts in ground_truth.items():
        pred_counts = predictions.get(image_id, {})
        for class_name in class_names:
            gt_count = counts.get(class_name, 0)
            pred_count = pred_counts.get(class_name, 0)
            diff = abs(pred_count - gt_count)
            sum_final += diff
            sum_sq_error[class_name] += diff ** 2
            sum_rel_sq_error[class_name] += (diff ** 2) / (gt_count + 1)
            if gt_count > 0:
                non_zero_count[class_name] += 1

    print("Sum of absolute differences:", sum_final)
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
