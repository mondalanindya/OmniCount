# import os
# import json
# from PIL import Image
# import torch
# import numpy as np
# from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation, ViTFeatureExtractor, ViTModel
# from sklearn.cluster import SpectralClustering
# from sklearn.metrics import silhouette_score

# # Initialize CLIPSeg and ViT models, feature extractors, and set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# clip_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
# clip_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
# vit_feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits8')
# vit_model = ViTModel.from_pretrained('facebook/dino-vits8')

# # Directory paths and vocabulary
# input_dir = str(OUTPUTS_ROOT / "FSC_test_split_multi" / "images")
# output_mask_dir = str(CLIPSEG_OUTPUTS_ROOT / "FSC_test_split_multi" / "images_clip")
# output_overlay_dir = str(CLIPSEG_OUTPUTS_ROOT / "FSC_test_split_multi" / "images_overlay")
# vocabulary = {"flower pots", "cashew nuts", "biscuits", "rice bags", "crab cakes", "peppers", "chairs", "skis", "shirts", "crayons", "milk cartons", "finger foods", "watches", "kidney beans", "jeans", "people", "marbles", "deers", "boxes", "sauce bottles", "potatoes", "cupcake tray", "green peas", "tree logs", "plates", "strawberries", "books", "suprmarket shelf", "potato chips", "sunglasses", "caps", "kiwis", "keyboard keys", "shoes", "cups", "bowls", "spoon", "flowers", "apples"}
# threshold = 0.5

# # Ensure output directories exist
# os.makedirs(output_mask_dir, exist_ok=True)
# os.makedirs(output_overlay_dir, exist_ok=True)

# # Dictionary to store results
# image_clusters = {}

# # Process each image for mask and overlay generation
# for filename in os.listdir(input_dir):
#     if filename.endswith(".jpg"):
#         # Load image
#         img_path = os.path.join(input_dir, filename)
#         original_image = Image.open(img_path).convert("RGB")

#         # Initialize dictionary for the current image
#         image_clusters[filename] = {}

#         # Predict and save masks for each class
#         for cls in vocabulary:
#             # Process inputs and move them to the device
#             inputs = clip_processor(text=cls, images=original_image, return_tensors="pt", padding=True)
#             inputs = {k: v.to(device) for k, v in inputs.items()}

#             # Get model output and process mask
#             outputs = clip_model(**inputs)
#             mask = outputs.logits.sigmoid()
#             binary_mask = mask > threshold
#             binary_mask_np = binary_mask.squeeze().cpu().numpy()

#             if np.any(binary_mask_np):
#                 # Create and save overlay image
#                 binary_mask_image = Image.fromarray(binary_mask_np.astype("uint8") * 255)
#                 binary_mask_image = binary_mask_image.resize(original_image.size).convert("L")
#                 overlay = Image.new("RGBA", original_image.size, (255, 255, 255, 0))
#                 colored_image = original_image.convert("RGBA")
#                 overlay.paste(colored_image, mask=binary_mask_image)
#                 overlay_dir = os.path.join(output_overlay_dir, filename.split('.')[0])
#                 os.makedirs(overlay_dir, exist_ok=True)
#                 overlay_image_path = os.path.join(overlay_dir, f"{cls}.png")
#                 overlay.save(overlay_image_path)

#                 # Apply spectral clustering to the overlay image
#                 _, optimal_clusters, _ = SpectralClustering(overlay_image_path, max_clusters=10)
#                 image_clusters[filename][cls] = optimal_clusters

# # Save the results to a JSON file
# with open('/vol/research/am04485/Codes/voting/outputs_clipseg/FSC_test_split_multi/fsc_multi_clipseg.json', 'w') as json_file:
#     json.dump(image_clusters, json_file, indent=4)


import os
import json
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, CLIPSegProcessor, CLIPSegForImageSegmentation, ViTFeatureExtractor, ViTModel
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import torch.nn.functional as F
from paths import OUTPUTS_ROOT, CLIPSEG_OUTPUTS_ROOT
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'
# Initialize models and processors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clip_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
vit_feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits8')
vit_model = ViTModel.from_pretrained('facebook/dino-vits8').to(device)

# Directory path and vocabulary
input_dir = str(OUTPUTS_ROOT / "FSC_test_split_multi" / "images")
vocabulary = {"flower pots", "cashew nuts", "biscuits", "rice bags", "crab cakes", "peppers", "chairs", "skis", "shirts", "crayons", "milk cartons", "finger foods", "watches", "kidney beans", "jeans", "people", "marbles", "deers", "boxes", "sauce bottles", "potatoes", "cupcake tray", "green peas", "tree logs", "plates", "strawberries", "books", "suprmarket shelf", "potato chips", "sunglasses", "caps", "kiwis", "keyboard keys", "shoes", "cups", "bowls", "spoon", "flowers", "apples"}
vocab_list = ["flower pots", "cashew nuts", "biscuits", "rice bags", "crab cakes", "peppers", "chairs", "skis", "shirts", "crayons", "milk cartons", "finger foods", "watches", "kidney beans", "jeans", "people", "marbles", "deers", "boxes", "sauce bottles", "potatoes", "cupcake tray", "green peas", "tree logs", "plates", "strawberries", "books", "suprmarket shelf", "potato chips", "sunglasses", "caps", "kiwis", "keyboard keys", "shoes", "cups", "bowls", "spoon", "flowers", "apples"]
threshold = 0.5

# Function to extract features using ViT
def extract_features(image):
    inputs = vit_feature_extractor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = vit_model(**inputs)
    return outputs.last_hidden_state.squeeze(0).cpu().numpy()

# Function for spectral clustering with silhouette score for cluster number selection
def spectral_clustering(features, max_clusters=10):
    best_score = -1
    best_num_clusters = 2
    for num_clusters in range(2, max_clusters + 1):
        clustering = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', n_init=10, random_state=0)
        labels = clustering.fit_predict(features)
        score = silhouette_score(features, labels)
        if score > best_score:
            best_score = score
            best_num_clusters = num_clusters
    return best_num_clusters

# Dictionary to store results
image_clusters = {}

for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        img_path = os.path.join(input_dir, filename)
        original_image = Image.open(img_path).convert("RGB")
        image_clusters[filename] = {}

        inputs = clip_processor(text=["flower pots", "cashew nuts", "biscuits", "rice bags", "crab cakes", "peppers", "chairs", "skis", "shirts", "crayons", "milk cartons", "finger foods", "watches", "kidney beans", "jeans", "people", "marbles", "deers", "boxes", "sauce bottles", "potatoes", "cupcake tray", "green peas", "tree logs", "plates", "strawberries", "books", "suprmarket shelf", "potato chips", "sunglasses", "caps", "kiwis", "keyboard keys", "shoes", "cups", "bowls", "spoon", "flowers", "apples"], images=[original_image]*len(vocab_list), return_tensors="pt", padding=True).to(device)
        h, w = inputs['pixel_values'].shape[-2:]
        fixed_scale = (512, 512)
        inputs['pixel_values'] = F.interpolate(
            inputs['pixel_values'],
            size=fixed_scale,
            mode='bilinear',
            align_corners=False)
        outputs = clip_model(**inputs)
        for hmap in range(0,len(vocab_list)):
            heatmaps = outputs.logits[hmap,:,:]
            mask = heatmaps.sigmoid() > 0.4
            if mask[mask == True].shape[0] > 0:
                mask_resized = Image.fromarray(mask.squeeze().cpu().numpy().astype('int32')).resize(original_image.size, Image.NEAREST)
                mask_resized_np = np.array(mask_resized)
                
                masked_image = np.array(original_image) * mask_resized_np[..., None]
                masked_image_pil = Image.fromarray(masked_image.astype(np.uint8)).convert("RGB")
                features = extract_features(masked_image_pil.resize((224, 224)))

                features_flattened = features.reshape(features.shape[0], -1)
                optimal_clusters = spectral_clustering(features_flattened)
                image_clusters[filename][vocab_list[hmap]] = optimal_clusters



        # # mask = outputs.logits.sigmoid() > 0.5
        # # print hello
        # for cls in vocabulary:
        #     inputs = clip_processor(text=cls, images=original_image, return_tensors="pt", padding=True).to(device)
        #     outputs = clip_model(**inputs)
        #     mask = outputs.logits.sigmoid() > 0.5
        #     mask_resized = Image.fromarray(mask.squeeze().cpu().numpy().astype('float32')).resize(original_image.size, Image.NEAREST)
        #     mask_resized_np = np.array(mask_resized) > 0.5

        #     masked_image = np.array(original_image) * mask_resized_np[..., None]
        #     masked_image_pil = Image.fromarray(masked_image.astype(np.uint8)).convert("RGB")
        #     features = extract_features(masked_image_pil.resize((224, 224)))

        #     # Flatten the features to apply clustering
        #     features_flattened = features.reshape(features.shape[0], -1)
        #     optimal_clusters = spectral_clustering(features_flattened)
        #     image_clusters[filename][cls] = optimal_clusters

with open(str(CLIPSEG_OUTPUTS_ROOT / "FSC_test_split_multi" / "fsc_multi_clipseg.json"), 'w') as json_file:
    json.dump(image_clusters, json_file, indent=4)

# print("Cluster data saved to image_clusters.json.")
