import torch
import cv2
from skimage.feature import peak_local_max
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
from tfoc.shi_segment_anything import sam_model_registry, SamPredictor
from tfoc.shi_segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from sam.segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator as SamAutomaticMaskGenerator1
from tfoc.utils import *
from transformers import AutoProcessor, CLIPSegVisionModel
from scipy.ndimage import gaussian_filter
from clips import clip
import os
from paths import OUTPUTS_ROOT, REPO_ROOT
import json

BICUBIC = InterpolationMode.BICUBIC

def refine_count(cls_id, patch_path, mask_path, target_texts, prompt_type, annotations, raw_image_list):
    if not os.path.exists(patch_path) or not os.path.exists(mask_path):
        print(f"Skipping {cls_id} as patch or mask does not exist.")
        return

    sam_anything = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/16", device=device)
    model.eval()
    preprocess = Compose([Resize((512, 512), interpolation=BICUBIC), ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    pil_img = Image.open(patch_path)
    raw_img = Image.open(raw_image_list)
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
    anno = annotations[cls_id+'.jpg']
    bboxes = anno['box_examples_coordinates']
    dots = np.array(anno['points'])

    if prompt_type == "text":
        if sam_anything:
            mask_generator = SamAutomaticMaskGenerator1(sam)
            masks = mask_generator.generate(np.asarray(pil_img))
        else:
            mask_generator = SamAutomaticMaskGenerator(sam)
            masks = mask_generator.generate(np.asarray(pil_img), flipped_coordinates)

        if len(masks) < 5:
            mask_generator = SamAutomaticMaskGenerator1(sam)
            masks = mask_generator.generate(np.asarray(pil_img))
    else:
        input_prompt = list()
        for bbox in bboxes:
            x1, y1 = bbox[0]
            x2, y2 = bbox[2]
            if prompt_type == 'box':
                input_prompt.append([x1, y1, x2, y2])
            elif prompt_type == 'point':
                input_prompt.append([(x1+x2)//2, (y1+y2)//2])

        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(np.asarray(raw_img), input_prompt)

    output_path = str(OUTPUTS_ROOT / "FSC_test_split_single" / "sam_patches_box_v2" / cls_id / target_texts)
    print("Creating output directory at: ", output_path)
    os.makedirs(output_path, exist_ok=True)

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

data_path = str(OUTPUTS_ROOT / "FSC_test_split_single")
anno_file = os.path.join(data_path, 'annotation_FSC147_384.json')
raw_img_path = str(OUTPUTS_ROOT / "FSC_test_split_single" / "images")

with open(anno_file) as f:
    annotations = json.load(f)

prompt_type = "box"

patch_path = os.path.join(data_path, "patches")
mask_path = os.path.join(data_path, "refined_bin_masks")

for cls_id in os.listdir(patch_path):
    patch_dir = os.path.join(patch_path, cls_id)
    mask_dir = os.path.join(mask_path, cls_id)
    raw_image_path = os.path.join(raw_img_path, cls_id + '.jpg')

    if not os.path.exists(mask_dir):
        print(f"No masks found for {cls_id}, skipping.")
        continue

    for patch_file in os.listdir(patch_dir):
        if not patch_file.startswith("patch_"):
            continue
        class_name = patch_file.split("patch_")[-1].split(".")[0]
        mask_file = f"mask_{class_name}.png"
        
        patch_full_path = os.path.join(patch_dir, patch_file)
        mask_full_path = os.path.join(mask_dir, mask_file)

        refine_count(cls_id, patch_full_path, mask_full_path, class_name, prompt_type, annotations, raw_image_path)
