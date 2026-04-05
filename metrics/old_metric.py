###################################OLD CODE############################################
# import numpy as np
# import json



# correct_class = 0 
# correct_count = 0
# count_per_class = 0
# img_list = list(gt_count.keys())
# tp = 0
# fp = 0
# tn = 0
# fn = 0
# def calculate(gt,pred):
#     correct_class = 0
#     correct_count = 0
#     count_per_class = 0
#     count_pred = []
#     count_gt =[]
#     if len(list(gt.keys())) == len(list(pred.keys())):
#         for classes in gt.keys(): ## classes = ['glass','bottle','grapes']
#             # print('gt class', classes)
#             if classes in pred.keys(): ## classes = ['glass','bottle','grapes'] 
#                     correct_class = 1
#                     # print('class is present in pred')
#                     if abs(gt[classes] - pred[classes]) <= 3: 
#                             count_per_class += 1
#                             # print('gt count', gt[classes])
#                             # print('pred count', pred[classes])
#                     else:
#                         correct_count = 0
#                         break
#             else:
#                 correct_class = 0
#                 for classes in gt.keys():
#                     count_gt.append(gt[classes])
#                 for classes in pred.keys():
#                     count_pred.append(pred[classes])
#                 # print('count_gt',count_gt)
#                 # print('count_pred',count_pred)
#                 for i in range(len(count_gt)):
#                     if abs(count_gt[i] - count_pred[i]) <= 3:
#                         # print(count_per_class)
#                         count_per_class += 1
                        
#                     else:
#                         correct_count = 0
#                         break
                
#                 break

#         if count_per_class == len(list(gt.keys())):
#             correct_count = 1
#     else:
#         correct_class = 0 

#     # print(correct_class, correct_count)

#     return correct_class, correct_count


# for img in img_list:
#     if img not in pred_count:
#         continue
#     gt = gt_count[img] ### gt = {'glass':1,'bottle':1,'grapes':36}
#     pred = pred_count[img] ### pred = {'glass':3,'bottle':2,'grapes':3}
#     # print(gt,pred)
#     # print('img',img)

#     correct_class, correct_count = calculate(gt,pred)
    
#     if correct_class == 1 and correct_count == 1:
#         tp+=1
#     # elif correct_class == 1 and correct_count == 0:
#         # tn+=1
#     elif correct_class == 0 and correct_count == 1:
#         fp+=1
#     elif correct_class == 0 and correct_count == 0:
#         fn+=1

#     print(tp,fp,tn,fn)

# # print('outside',tp,fp,tn,fn)
# precision = tp/(tp+fp)
# print('precision: ',precision)
# recall = tp/(tp+fn)
# print('recall: ',recall)
# f1 = 2*(precision*recall)/(precision+recall)
# print('f1: ',f1)
# # acc = (tp+tn)/(tp+tn+fp+fn)
# # print('accuracy: ',acc)

# # print('precision: ',precision)
# # print('recall: ',recall)
############################OLD CODE ENDS############################################

###################################CODE v2############################################

import json

from paths import OUTPUTS_ROOT, REPO_ROOT

# Load ground truth and predictions
with open(OUTPUTS_ROOT / 'FSC_test_split_multi' / 'fsc_data_gt.json') as f:
    gt_count = json.load(f)

with open(REPO_ROOT / 'outputs_clipseg' / 'FSC_test_split_multi' / 'fsc_multi_clipseg.json') as f:
    pred_count = json.load(f)

# Initialize counters
tp = 0
fp = 0
fn = 0

def calculate(gt, pred):
    correct_class = 0
    correct_count = 0
    
    # Check for correct class presence
    if set(gt.keys()) == set(pred.keys()):
        correct_class = 1
        # Now check each class count within tolerance
        for class_name in gt.keys():
            if abs(gt[class_name] - pred[class_name]) <= 5:
                correct_count += 1
            else:
                # If any class count is not within tolerance, stop checking further
                correct_count = 0
                break
    else:
        correct_class = 0

    # If all classes' counts are within tolerance, consider it correct
    if correct_count == len(gt.keys()):
        correct_count = 1
    else:
        correct_count = 0

    return correct_class, correct_count

# Process each image
for img in gt_count.keys():
    if img not in pred_count:
        # If an image in GT is not found in predictions, consider it as FN
        fn += 1
        continue

    gt = gt_count[img]
    pred = pred_count[img]

    correct_class, correct_count = calculate(gt, pred)
    
    # Update TP, FP, FN based on class and count correctness
    if correct_class == 1 and correct_count == 1:
        tp += 1
    elif correct_class == 1 and correct_count == 0:
        fn += 1  # Correct class but incorrect count
    elif correct_class == 0:
        fp += 1  # Incorrect class

# Calculate metrics
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

###################################CODE v2 ends############################################


