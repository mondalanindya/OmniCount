# import torch
# import cv2
# from skimage.feature import peak_local_max
# import sys
# import numpy as np
# from PIL import Image
# from  matplotlib import pyplot as plt
# import torch
# import torch.nn.functional as F
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize
# from torchvision.transforms import InterpolationMode
# BICUBIC = InterpolationMode.BICUBIC
# from tfoc.shi_segment_anything import sam_model_registry, SamPredictor
# from tfoc.shi_segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
# from sam.segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator as SamAutomaticMaskGenerator1
# from tfoc.utils import *
# import requests
# from transformers import AutoProcessor, CLIPSegVisionModel
# from scipy.ndimage import gaussian_filter
# from clips import clip
# import argparse
# import os





# def refine_count(cls_id,patch_path,mask_path,target_texts):
#     if not os.path.exists(patch_path) or not os.path.exists(mask_path):
#         print(f"Patch or mask not found for {cls_id}")
#         return


#     sam_anything = False
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, _ = clip.load("ViT-B/16", device=device)
#     model.eval()
#     #low dim preprocess
#     # preprocess =  Compose([Resize((224, 224), interpolation=BICUBIC), ToTensor(),
#     #     Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
#     #high dim preprocess
#     preprocess =  Compose([Resize((512, 512), interpolation=BICUBIC), ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


#     pil_img = Image.open(patch_path)
#     cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
#     image = preprocess(pil_img).unsqueeze(0).to(device)
#     all_texts = ['airplane', 'bag', 'strawberries', 'bedclothes', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'horse', 'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform', 'potted plant', 'road', 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table', 'track', 'train', 'tree', 'truck', 'tv monitor', 'wall', 'water', 'window', 'wood']
#     target_texts = target_texts




#     model, preprocess = clip.load("CS-ViT-B/16", device=device)
#     model.eval()

#     with torch.no_grad():
        
#         image_features = model.encode_image(image)
#         image_features = image_features / image_features.norm(dim=1, keepdim=True)
#         text_features = clip.encode_text_with_prompt_ensemble(model, all_texts, device)
#         similarity = clip.clip_feature_surgery(image_features, text_features)
#         similarity_map = clip.get_similarity_map(similarity[:, 1:, :], cv2_img.shape[:2])

        
#         # for b in range(similarity_map.shape[0]):
#         #     for n in range(similarity_map.shape[-1]):
#         #         if all_texts[n] not in target_texts:
#         #             continue
#         #         vis = (similarity_map[b, :, :, n].cpu().numpy() * 255).astype('uint8')
#         #         vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
#         #         vis = cv2_img * 0.4 + vis * 0.6
#         #         vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)



#     processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
#     model = CLIPSegVisionModel.from_pretrained("CIDAS/clipseg-rd64-refined")

#     inputs = processor(images=pil_img, return_tensors="pt")

#     outputs = model(**inputs)
#     last_hidden_state = outputs.last_hidden_state


#     pred = np.array(Image.open(mask_path))



#     pooled_hidden_state = F.adaptive_avg_pool2d(last_hidden_state, pred.shape)
#     binary_mask_tensor = torch.from_numpy(pred).float()
#     expanded_mask = binary_mask_tensor.unsqueeze(0)
#     result = pooled_hidden_state * expanded_mask
#     result.dtype

#     new_mask = similarity_map.mean(dim=3).squeeze(0)
#     new_mask = new_mask.cpu().detach().numpy()
#     exp_mask = expanded_mask.cpu().detach().numpy()
#     exp_mask1 = exp_mask/255.0

#     new_mask1 = 1-new_mask
#     np.max(new_mask1)
#     hadamard = new_mask1 * exp_mask1
#     # hmap_hidden_state = pooled_hidden_state.cpu().detach().numpy()
#     org_res = hadamard * new_mask1
#     org_res.shape


#     hadamard = hadamard.squeeze()  
#     smoothed = cv2.GaussianBlur(hadamard, (5, 5), sigmaX=0)
#     # def taylor_refinement(smoothed, coordinates):
#     #     updated_coordinates = []
#     #     for coord in coordinates:
#     #         px, py = int(coord[0]), int(coord[1])
#     #         if 1 < px < smoothed.shape[1] - 2 and 1 < py < smoothed.shape[0] - 2:
#     #             dx = 0.5 * (smoothed[py][px+1] - smoothed[py][px-1])
#     #             dy = 0.5 * (smoothed[py+1][px] - smoothed[py-1][px])
#     #             dxx = 0.25 * (smoothed[py][px+2] - 2 * smoothed[py][px] + smoothed[py][px-2])
#     #             dxy = 0.25 * (smoothed[py+1][px+1] - smoothed[py-1][px+1] - smoothed[py+1][px-1] + smoothed[py-1][px-1])
#     #             dyy = 0.25 * (smoothed[py+2][px] - 2 * smoothed[py][px] + smoothed[py-2][px])
#     #             derivative = np.matrix([[dx], [dy]])
#     #             hessian = np.matrix([[dxx, dxy], [dxy, dyy]])
#     #             if np.linalg.det(hessian) != 0:  # Check if the determinant of the hessian is not zero
#     #                 hessianinv = np.linalg.inv(hessian)  # Use np.linalg.inv() for inverse
#     #                 offset = -hessianinv * derivative
#     #                 offset = np.squeeze(np.asarray(offset.T), axis=0)
#     #                 # Convert coord to float for addition, then round or cast back to int if necessary
#     #                 coord_float = coord.astype(np.float64)
#     #                 coord_updated = coord_float + offset
#     #                 # If you need the coordinates as integers (e.g., for indexing), round them
#     #                 coord_updated = np.round(coord_updated).astype(np.int64)
#     #         updated_coordinates.append(coord_updated)
#     #     return np.array(updated_coordinates)
#     coordinates = peak_local_max(smoothed, min_distance=4, threshold_rel=0.8)
#     # updated_coordinates = taylor_refinement(smoothed, coordinates)
#     flipped_coordinates = [[y, x] for x, y in coordinates]

#     if len(flipped_coordinates) == 0 or len(flipped_coordinates) < 5:
#         sam_anything = True

    


#     sam_checkpoint = "sam_vit_b_01ec64.pth"
#     model_type = "vit_b"
#     device = "cuda"
#     sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#     sam.to(device=device)
#     if sam_anything:
#         mask_generator = SamAutomaticMaskGenerator1(sam)
#         masks = mask_generator.generate(np.asarray(pil_img))
#     else:
#         mask_generator = SamAutomaticMaskGenerator(sam)
#         masks = mask_generator.generate(np.asarray(pil_img), flipped_coordinates)


#     if len(masks) < 5:
#         mask_generator = SamAutomaticMaskGenerator1(sam)
#         masks = mask_generator.generate(np.asarray(pil_img))

#     # Create output directory

    

#     print("Creating output directory at: ", output_path)
#     os.makedirs(output_path, exist_ok=True)

#     # Overlay masks and save
#     def overlay_and_save(image, mask, output_path, filename):
#         overlayed_image = np.where(mask[..., None], image, np.zeros_like(image))
#         cv2.imwrite(os.path.join(output_path, filename), cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR))

#     for i, mask_dict in enumerate(masks):
#         mask = mask_dict.get('segmentation', None)
#         if mask is None:
#             continue

#         if len(mask.shape) > 2:
#             mask = mask[:, :, 0]

#         filename = f'{target_texts}_{i}.png'
#         overlay_and_save(np.asarray(pil_img), mask, output_path, filename)
#         print(f"Overlayed mask saved at {output_path}/{filename}")




# patches = os.listdir(patch_path)
# masks = os.listdir(mask_path)
# cls_name_list =[]
# for cls_id in patches:
#     # print('Now processing image:', cls_id)
#     patch_list = os.path.join(patch_path, cls_id)
#     mask_list = os.path.join(mask_path, cls_id)
#     # if os.path.exists(os.path.join(output_path,cls_id)):
#     #     # os.makedirs(os.path.join(output_path,cls_id), exist_ok=True)
#     #     continue
#     # else:
#     print("Current processing:", cls_id)
    
#     # os.makedirs(os.path.join(output_path,cls_id), exist_ok=True)
#     for cls_name in os.listdir(mask_list):

        
#         cls_name_split = os.path.join(mask_list, cls_name)
#         cls_name_sep = cls_name.split('.')[0].split('_')[-1]
#         patch_name_list = os.path.join(patch_list, 'patch_'+ cls_name_sep +'.png')
#         mask_name_list = os.path.join(mask_list, 'mask_'+ cls_name_sep +'.png')
#         # print(cls_name_sep)
#         refine_count(cls_id,patch_name_list,mask_name_list,cls_name_sep)
        





import torch
import cv2
from skimage.feature import peak_local_max
import sys
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from tfoc.shi_segment_anything import sam_model_registry, SamPredictor
from tfoc.shi_segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from sam.segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator as SamAutomaticMaskGenerator1
from tfoc.utils import *
from transformers import AutoProcessor, CLIPSegVisionModel
from scipy.ndimage import gaussian_filter
from clips import clip
import os
from paths import OUTPUTS_ROOT, REPO_ROOT
sam_checkpoint = str(REPO_ROOT / "sam_vit_b_01ec64.pth")

def refine_count(cls_id, patch_path, mask_path, target_texts):
    # Check if the patch and mask exist, if not, skip
    if not os.path.exists(patch_path) or not os.path.exists(mask_path):
        print(f"Skipping {cls_id} as patch or mask does not exist.")
        return

    sam_anything = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/16", device=device)
    model.eval()
    preprocess = Compose([Resize((512, 512), interpolation=BICUBIC), ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    pil_img = Image.open(patch_path)
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    image = preprocess(pil_img).unsqueeze(0).to(device)
    all_texts = ['airplane', 'bag', 'strawberries', 'bedclothes', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'horse', 'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform', 'potted plant', 'road', 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table', 'track', 'train', 'tree', 'truck', 'tv monitor', 'wall', 'water', 'window', 'wood']
    target_texts = target_texts

    model, preprocess = clip.load("CS-ViT-B/16", device=device)
    model.eval()

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = clip.encode_text_with_prompt_ensemble(model, all_texts, device)
        similarity = clip.clip_feature_surgery(image_features, text_features)
        similarity_map = clip.get_similarity_map(similarity[:, 1:, :], cv2_img.shape[:2])

    processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegVisionModel.from_pretrained("CIDAS/clipseg-rd64-refined")

    inputs = processor(images=pil_img, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state

    pred = np.array(Image.open(mask_path))

    pooled_hidden_state = F.adaptive_avg_pool2d(last_hidden_state, pred.shape)
    binary_mask_tensor = torch.from_numpy(pred).float()
    expanded_mask = binary_mask_tensor.unsqueeze(0)
    result = pooled_hidden_state * expanded_mask
    result.dtype

    new_mask = similarity_map.mean(dim=3).squeeze(0)
    new_mask = new_mask.cpu().detach().numpy()
    exp_mask = expanded_mask.cpu().detach().numpy()
    exp_mask1 = exp_mask/255.0

    new_mask1 = 1-new_mask
    np.max(new_mask1)
    hadamard = new_mask1 * exp_mask1
    org_res = hadamard * new_mask1
    org_res.shape

    hadamard = hadamard.squeeze()
    smoothed = cv2.GaussianBlur(hadamard, (5, 5), sigmaX=0)
    coordinates = peak_local_max(smoothed, min_distance=4, threshold_rel=0.8)
    flipped_coordinates = [[y, x] for x, y in coordinates]

    if len(flipped_coordinates) == 0 or len(flipped_coordinates) < 5:
        sam_anything = True

    sam_checkpoint = str(REPO_ROOT / "sam_vit_b_01ec64.pth")
    model_type = "vit_b"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    if sam_anything:
        mask_generator = SamAutomaticMaskGenerator1(sam)
        masks = mask_generator.generate(np.asarray(pil_img))
    else:
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(np.asarray(pil_img), flipped_coordinates)

    if len(masks) < 5:
        mask_generator = SamAutomaticMaskGenerator1(sam)
        masks = mask_generator.generate(np.asarray(pil_img))

    # Create output directory
    output_path = str(OUTPUTS_ROOT / "FSC_test_split_single" / "sam_patches_64" / cls_id / target_texts)
    print("Creating output directory at: ", output_path)
    os.makedirs(output_path, exist_ok=True)

    # Overlay masks and save
    def overlay_and_save(image, mask, output_path, filename):
        overlayed_image = np.where(mask[..., None], image, np.zeros_like(image))
        cv2.imwrite(os.path.join(output_path, filename), cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR))

    for i, mask_dict in enumerate(masks):
        mask = mask_dict.get('segmentation', None)
        if mask is None:
            continue

        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        filename = f'{target_texts}_{i}.png'
        overlay_and_save(np.asarray(pil_img), mask, output_path, filename)
        print(f"Overlayed mask saved at {output_path}/{filename}")

patch_path = str(OUTPUTS_ROOT / "FSC_test_split_single" / "patches")
mask_path = str(OUTPUTS_ROOT / "FSC_test_split_single" / "refined_bin_masks")
output_path = str(OUTPUTS_ROOT / "FSC_test_split_single" / "sam_patches_64")

patches = os.listdir(patch_path)
masks = os.listdir(mask_path)
cls_name_list = []
for cls_id in patches:
    print("Current processing:", cls_id)
    patch_list = os.path.join(patch_path, cls_id)
    mask_list = os.path.join(mask_path, cls_id)
    if not os.path.exists(patch_list) or not os.path.exists(mask_list):
        print(f"No patches or masks found for {cls_id}, skipping.")
        continue
    for cls_name in os.listdir(mask_list):
        cls_name_split = os.path.join(mask_list, cls_name)
        cls_name_sep = cls_name.split('.')[0].split('_')[-1]
        patch_name_list = os.path.join(patch_list, 'patch_' + cls_name_sep + '.png')
        mask_name_list = os.path.join(mask_list, 'mask_' + cls_name_sep + '.png')
        refine_count(cls_id, patch_name_list, mask_name_list, cls_name_sep)
