import os
import glob
import json

from PIL import Image
import numpy as np

from paths import OUTPUTS_ROOT

# def get_classes_in_directory(directory):
#     return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

# def load_image(img_path, resize=None, pil=False):
#     image = Image.open(img_path).convert("RGB")
#     if resize is not None:
#         image = image.resize((resize, resize))
#     if pil:
#         return image
#     return np.asarray(image).astype(np.float32) / 255.


# def process_images(base_directory, save_directory, threshold=0.5):
#     # clip_model = "ViT-B/32" #@param ["RN50", "RN101", "RN50x4", "RN50x16", "ViT-B/32", "ViT-B/16"]
#     # # image_caption = 'the dog' #@param {type:"string"}
#     # topk =  1#@param {type:"integer"}

#     # Load CLIP model.
#     # device = "cuda" if torch.cuda.is_available() else "cpu"
#     # model, preprocess = clip.load(clip_model, device=device, jit=False)
#     model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
#     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#     for img_folder in os.listdir(base_directory):
#         folder_path = os.path.join(base_directory, img_folder)
#         if os.path.isdir(folder_path):
#             class_counts = {}
#             for class_name in get_classes_in_directory(folder_path):
#                 class_counts[class_name] = 0
#                 class_path = os.path.join(folder_path, class_name)
#                 # print(class_path)
#                 other_classes = [c for c in get_classes_in_directory(folder_path) if c != class_name]
#                 # print(get_classes_in_directory(folder_path))
#                 # other_classes.append('black')
#                 # other_classes.append('background')

#                 # print(other_classes)
#                 prompt_list = []
#                 if len(other_classes) == 0:
#                     prompt_list.append(f"a photo of a "+class_name+".")
#                     prompt_list.append(f"a photo of black color.")
#                     # prompt_list.append(f"not a photo of "+class_name+".")
#                 else:
#                     for filtered_cls in other_classes:
#                         prompt = f"not a photo of a "+filtered_cls+", background."
#                         prompt_list.append(prompt)
#                         # prompt_list.append(f"a photo of a "+class_name+".")
#                         # prompt_list.append(f"a photo of black color.")

#                 # text_input = clip.tokenize([prompt]).to(device)
#                 # print(prompt_list)
                
#                 for img_file in os.listdir(class_path):
#                     if img_file.endswith(('.png', '.jpg', '.jpeg')):
#                         img_path = os.path.join(class_path, img_file)
#                         # print(img_path)
#                         # img_tensor = transforms.ToTensor()(load_image(img_path, True))
#                         # p = model.visual.input_resolution - 224
#                         # patches_pad = torch.nn.functional.pad(
#                         #     img_tensor, (p//2, p//2, p//2, p//2), "constant", 0).to(device)
#                         image = Image.open(img_path).convert("RGB")

#                         # with torch.no_grad():
#                         #     patch_embs = model.encode_image(patches_pad)
#                         #     text_embs = model.encode_text(text_input)
#                         #     patch_embs = patch_embs / patch_embs.norm(dim=-1, keepdim=True)
#                         #     text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
#                         #     sim = patch_embs @ text_embs.t()
#                             # idx_max = sim.argmax().item()
#                             # print(sim)


#                         inputs = processor(text=prompt_list, images=image, return_tensors="pt", padding=True).to("cuda")
#                         outputs = model(**inputs)
#                         logits_per_image = outputs.logits_per_image
#                         probs = logits_per_image.softmax(dim=1)

#                         # print(probs)
#                         # logits_per_image = torch.nn.functional.softmax(logits_per_image, dim=1)
#                         # print(logits_per_image)
#                         # print(logits_per_image.max().item())
#                         if probs.max().item() > threshold : #### +ve class score
#                         #     continue
#                         # else:
#                             class_counts[class_name] += 1
#                             # Save images with lowest match
#                             save_img_path = img_path.replace(base_directory, save_directory)
#                             os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
#                             image.save(save_img_path)

#             for class_name, count in class_counts.items():
#                 print(f"For {img_folder}.jpg, the count of {class_name} = {count}")

from transformers import CLIPProcessor, CLIPModel

# Define all classes
# all_classes = {"person", "sea shells", "balloons", "peppers", "buns", "strawberries", "bowls", "cups", "plates", "grapes", "birds", "bicycle", "eggs", "books", "sheep"}

# def process_images(base_directory, save_directory, threshold=0.4):
#     # Load CLIP model
#     model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
#     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#     for img_folder in os.listdir(base_directory):
#         folder_path = os.path.join(base_directory, img_folder)
#         if os.path.isdir(folder_path):
#             class_counts = {}
#             for class_name in os.listdir(folder_path):
#                 if os.path.isdir(os.path.join(folder_path, class_name)):
#                     class_counts[class_name] = 0
#                     class_path = os.path.join(folder_path, class_name)

#                     # Generate prompt list excluding the current class
#                     prompt_list = [f"{c}" for c in all_classes if c != class_name]
#                     # prompt_list.append("background")

#                     for img_file in os.listdir(class_path):
#                         if img_file.endswith(('.png', '.jpg', '.jpeg')):
#                             img_path = os.path.join(class_path, img_file)
#                             image = Image.open(img_path).convert("RGB")

#                             inputs = processor(text=prompt_list, images=image, return_tensors="pt", padding=True).to("cuda")
#                             outputs = model(**inputs)
#                             logits_per_image = outputs.logits_per_image
#                             probs = logits_per_image.softmax(dim=1)
#                             # print(probs)

#                             if probs.max().item() > threshold:
#                                 class_counts[class_name] += 1
#                                 save_img_path = img_path.replace(base_directory, save_directory)
#                                 os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
#                                 image.save(save_img_path)

#             for class_name, count in class_counts.items():
#                 print(f"For {img_folder}.jpg, the count of {class_name} = {count}")


##############################################################################################################

# import os
# from PIL import Image
# import torch
# from transformers import CLIPProcessor, CLIPModel

# def get_classes_in_directory(directory):
#     return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

# def process_images(base_directory, save_directory, threshold=0.8):
#     all_classes = {"person", "sea shells", "baloons", "peppers", "buns", "strawberries", "bowls", "cups", "plates", "grapes", "birds", "bicycle", "eggs", "books", "sheep"}
#     model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
#     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#     for img_folder in os.listdir(base_directory):
#         folder_path = os.path.join(base_directory, img_folder)
#         if os.path.isdir(folder_path):
#             class_counts = {}
#             for class_name in get_classes_in_directory(folder_path):
#                 class_counts[class_name] = 0
#                 class_path = os.path.join(folder_path, class_name)

#                 prompt_list = [f"{c}" for c in all_classes if c != class_name]
#                 prompt_list.append("background")

#                 for img_file in os.listdir(class_path):
#                     if img_file.endswith(('.png', '.jpg', '.jpeg')):
#                         img_path = os.path.join(class_path, img_file)
#                         image = Image.open(img_path).convert("RGB")

#                         inputs = processor(text=prompt_list, images=image, return_tensors="pt", padding=True).to("cuda")
#                         outputs = model(**inputs)
#                         logits_per_image = outputs.logits_per_image
#                         probs = logits_per_image.softmax(dim=1)
#                         top_probs, top_labels = torch.topk(probs, 2)

#                         assigned_class = prompt_list[top_labels[0][0]]
#                         print(f"Image: {img_file}, Assigned class: {assigned_class}, Prob: {top_probs[0][0].item()}")

#                         if top_probs[0][0].item() > threshold:
#                             if assigned_class == "background" and top_probs[0][1].item() > threshold:
#                                 assigned_class = prompt_list[top_labels[0][1]]
#                                 print(f"Reassigned class: {assigned_class}")
#                             if assigned_class == class_name:
#                                 class_counts[class_name] += 1
#                                 save_img_path = img_path.replace(base_directory, save_directory)
#                                 os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
#                                 image.save(save_img_path)
#                                 print(f"Saved: {save_img_path}")

#             for class_name, count in class_counts.items():
#                 print(f"For {img_folder}.jpg, the count of {class_name} = {count}")




# import os
# import json
# from PIL import Image
# import torch
# from transformers import CLIPProcessor, CLIPModel

# def process_images(base_directory, save_directory, threshold=0.3):
#     # Load CLIP model
#     model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
#     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#     # Dictionary to hold image counts
#     image_counts = {}

#     for img_folder in os.listdir(base_directory):
#         folder_path = os.path.join(base_directory, img_folder)
#         if os.path.isdir(folder_path):
#             class_counts = {}
#             for class_name in os.listdir(folder_path):
#                 if os.path.isdir(os.path.join(folder_path, class_name)):
#                     class_counts[class_name] = 0
#                     class_path = os.path.join(folder_path, class_name)

#                     # Generate prompt list excluding the current class
#                     prompt_list = [f"{c}" for c in all_classes if c != class_name]

#                     for img_file in os.listdir(class_path):
#                         if img_file.endswith(('.png', '.jpg', '.jpeg')):
#                             img_path = os.path.join(class_path, img_file)
#                             image = Image.open(img_path).convert("RGB")

#                             inputs = processor(text=prompt_list, images=image, return_tensors="pt", padding=True).to("cuda")
#                             outputs = model(**inputs)
#                             logits_per_image = outputs.logits_per_image
#                             probs = logits_per_image.softmax(dim=1)

#                             if probs.max().item() > threshold:
#                                 class_counts[class_name] += 1
#                                 save_img_path = img_path.replace(base_directory, save_directory)
#                                 os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
#                                 image.save(save_img_path)

#             image_counts[f"{img_folder}.jpg"] = class_counts

#     # Write to JSON file
#         json.dump(image_counts, json_file, indent=4)

# # Define all classes
# all_classes = {"bike", "biker", "bus", "car", "motorbike", "pedestrian", "trafficlight", "truck"}



##save the class counts from sam_patches to json file
# import os
# import json
# from PIL import Image
# import torch
# from transformers import CLIPProcessor, CLIPModel

# def save_json(root, json_file):
#     image_counts = {}
#     for img_folder in os.listdir(root):
#         folder_path = os.path.join(root, img_folder)
#         if os.path.isdir(folder_path):
#             class_counts = {}
#             for class_name in os.listdir(folder_path):
#                 if os.path.isdir(os.path.join(folder_path, class_name)):
#                     class_counts[class_name] = 0
#                     class_path = os.path.join(folder_path, class_name)
#                     for img_file in os.listdir(class_path):
#                         if img_file.endswith(('.png', '.jpg', '.jpeg')):
#                             class_counts[class_name] += 1
#             image_counts[f"{img_folder}.jpg"] = class_counts
#     with open(json_file, 'w') as json_file:
#         json.dump(image_counts, json_file, indent=4)




# import os
# import json
# from PIL import Image
# import torch
# from transformers import CLIPProcessor, CLIPModel

# def process_images(base_directory, save_directory):
#     model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
#     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#     image_counts = {}

#     for img_folder in os.listdir(base_directory):
#         folder_path = os.path.join(base_directory, img_folder)
#         if os.path.isdir(folder_path):
#             class_counts = {}
#             class_name_list = [f"{c}" for c in os.listdir(folder_path)]
#             prompt_init = ["background"]
#             prompt_list = [f"{c}" for c in class_name_list]
#             prompt_init.extend(prompt_list)
#             print(prompt_init)
#             for class_name in os.listdir(folder_path):
            
#                 class_dir = os.path.join(folder_path, class_name)
#                 if os.path.isdir(class_dir):
#                     class_counts[class_name] = 0
#                     for img_file in os.listdir(class_dir):
#                         if img_file.endswith(('.png', '.jpg', '.jpeg')):
#                             img_path = os.path.join(class_dir, img_file)
#                             image = Image.open(img_path).convert("RGB")

#                             inputs = processor(text=prompt_init, images=image, return_tensors="pt", padding=True).to("cuda")
#                             outputs = model(**inputs)
#                             logits_per_image = outputs.logits_per_image
#                             probs = logits_per_image.softmax(dim=1)
#                             # print(probs)
#                             probs_arr = probs.cpu().detach().numpy()
#                             # print(probs_arr)
#                             probs_arr_new = probs_arr[0][1:]
#                             # print(probs_arr_new)

#                             max_prob = probs_arr_new.max()
#                             max_prob_idx = probs_arr_new.argmax()

#                             if prompt_init[max_prob_idx+1] == class_name and max_prob > 0.5:
#                             # if max_prob > 0.5:
#                                 class_counts[class_name] += 1
#                                 save_img_path = img_path.replace(base_directory, save_directory)
#                                 os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
#                                 image.save(save_img_path)
#                             # top_probs, top_labels = torch.topk(probs, 2)

#                             # Get the class names for the top 2 predictions
#                             # top_classes = [prompt_list[idx] for idx in top_labels[0]]

#                             # # Check if the highest score is for "background"
#                             # if top_classes[0] == 'background':
#                             #     second_best_class = top_classes[1]
#                             #     if second_best_class == class_name:
#                             #         class_counts[class_name] += 1
#                             #         save_img_path = img_path.replace(base_directory, save_directory)
#                             #         os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
#                             #         image.save(save_img_path)

#             image_counts[f"{img_folder}.png"] = class_counts

#         json.dump(image_counts, json_file, indent=4)

# # Define all classes

# #all_classes = {"background", "flower pots", "cashew nuts", "biscuits", "rice bags", "crab cakes", "peppers", "chairs", "skis", "shirts", "crayons", "milk cartons", "finger foods", "watches", "kidney beans", "jeans", "people", "marbles", "deers", "boxes", "sauce bottles", "potatoes", "cupcake tray", "green peas", "tree logs", "plates", "strawberries", "books", "suprmarket shelf", "potato chips", "sunglasses", "caps", "kiwis", "keyboard keys", "shoes", "cups", "bowls", "spoon", "flowers", "apples"}
# #all classes should only contain the folder names under each image in the sam_patches folder



# import os
# import json
# from PIL import Image
# import torch
# from transformers import CLIPProcessor, CLIPModel
# import inflect

# def process_images(base_directory, save_directory):
#     model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
#     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#     p = inflect.engine()  # Initialize inflect engine

#     image_counts = {}

#     for img_folder in os.listdir(base_directory):
#         folder_path = os.path.join(base_directory, img_folder)
#         if os.path.isdir(folder_path):
#             class_counts = {}
#             class_name_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
#             prompt_init = ["background"]  # Initialize prompt list with background in desired format
#             prompt_list = []  # This will hold our modified class names for CLIP in the desired format
            
#             for class_path in class_name_list:
#                 class_name = os.path.basename(class_path)
#                 singular_form = p.singular_noun(class_name) or class_name
#                 formatted_class_name = f"this is a photo of a {singular_form}"
#                 prompt_list.append(formatted_class_name)
            
#             prompt_init.extend(prompt_list)
#             print(prompt_init)
            
#             for class_name in os.listdir(folder_path):
#                 class_dir = os.path.join(folder_path, class_name)
#                 if os.path.isdir(class_dir):
#                     class_counts[class_name] = 0
#                     singular_class_name = p.singular_noun(class_name) or class_name  # Convert to singular form for CLIP comparison
#                     formatted_class_name = f"this is a photo of a {singular_class_name}"  # Format for CLIP
                    
#                     for img_file in os.listdir(class_dir):
#                         if img_file.endswith(('.png', '.jpg', '.jpeg')):
#                             
                            
#                             img_path = os.path.join(class_dir, img_file)
                            
#                             # Instead of loading the image here, directly proceed to condition checks based on metadata or filenames if applicable

#                             # Load and process the image only if conditions are met (after inference, which requires an initial load)
#                             image = Image.open(img_path).convert("RGB")

#                             inputs = processor(text=prompt_init, images=image, return_tensors="pt", padding=True).to("cuda")
#                             outputs = model(**inputs)
#                             logits_per_image = outputs.logits_per_image
#                             probs = logits_per_image.softmax(dim=1)
#                             probs_arr = probs.cpu().detach().numpy()
#                             probs_arr_new = probs_arr[0][1:]  # Skip background

#                             max_prob = probs_arr_new.max()
#                             max_prob_idx = probs_arr_new.argmax()

#                             if prompt_init[max_prob_idx+1] == formatted_class_name and max_prob > 0.0:
#                                 class_counts[class_name] += 1  # Use original class name for counting
#                                 save_img_path = img_path.replace(base_directory, save_directory)
#                                 os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
#                                 image.save(save_img_path)

#             image_counts[f"{img_folder}.jpg"] = class_counts

#         json.dump(image_counts, json_file, indent=4)









###############################################################################################################################################


#single label count


# import os
# import glob
# import json
# from PIL import Image
# import numpy as np

# result = {}

# # Iterate over all directories in the base directory
# for dir_name in os.listdir(base_dir):
#     dir_path = os.path.join(base_dir, dir_name)
    
#     # Check if it's a directory
#     if os.path.isdir(dir_path):
#         for class_name in os.listdir(dir_path):
#             class_path = os.path.join(dir_path, class_name)
            
#             # Check if it's a directory
#             if os.path.isdir(class_path):
#                 image_files = glob.glob(os.path.join(class_path, "*.png"))
#                 image_count = 0
                
#                 # Check each image
#                 for image_file in image_files:
#                     image = np.array(Image.open(image_file))
                    
#                     # Check if all pixel values are <= 10
#                     if not np.all(image <= 5):
#                         image_count += 1
#                         # print(image_count)
                
#                 # Add to result
#                 result[f"{dir_name}.jpg"] = {class_name: image_count}

# # Write to JSON file
#     json.dump(result, f, indent=4)




########################################################################################################################

#multilabel count


# import os
# import glob
# import json
# from PIL import Image
# import numpy as np

# result = {}

# # Iterate over all directories in the base directory
# for dir_name in os.listdir(base_dir):
#     dir_path = os.path.join(base_dir, dir_name)
    
#     # Check if it's a directory
#     if os.path.isdir(dir_path):
#         image_dict = {}  # Initialize an empty dictionary for this image id
#         for class_name in os.listdir(dir_path):
#             class_path = os.path.join(dir_path, class_name)
            
#             # Check if it's a directory
#             if os.path.isdir(class_path):
#                 image_files = glob.glob(os.path.join(class_path, "*.png"))
#                 image_count = 0
                
#                 # Check each image
#                 for image_file in image_files:
#                     image = np.array(Image.open(image_file))
                    
#                     # Check if all pixel values are <= 5
#                     # if not np.all(image <= 10):
#                     image_count += 1
                
#                 # Add to image id's dictionary
#                 image_dict[class_name] = image_count

#         # Add to result
#         result[f"{dir_name}.jpg"] = image_dict

# # Write to JSON file
#     json.dump(result, f, indent=4)

import os
import glob
import json
from PIL import Image
import numpy as np

base_dir = str(OUTPUTS_ROOT / 'animals' / 'sam_patches')
result = {}

# Iterate over all directories in the base directory
for dir_name in os.listdir(base_dir):
    dir_path = os.path.join(base_dir, dir_name)
    
    # Check if it's a directory
    if os.path.isdir(dir_path):
        image_dict = {}  # Initialize an empty dictionary for this image id
        for class_name in os.listdir(dir_path):
            class_path = os.path.join(dir_path, class_name)
            
            # Check if it's a directory
            if os.path.isdir(class_path):
                image_files = glob.glob(os.path.join(class_path, "*.png"))
                total_pixel_values = 0
                image_count = 0
                
                # First pass: calculate total pixel values and number of images
                for image_file in image_files:
                    image = np.array(Image.open(image_file))
                    total_pixel_values += np.mean(image)
                    image_count += 1
                
                if image_count > 0:
                    average_pixel_value = total_pixel_values / image_count
                else:
                    average_pixel_value = 0
                
                # Second pass: count images with pixel value greater than the average
                qualifying_image_count = 0
                for image_file in image_files:
                    image = np.array(Image.open(image_file))
                    if np.mean(image) > 1*average_pixel_value:
                        qualifying_image_count += 1
                
                # Add to image id's dictionary only if the count is more than 0
                if qualifying_image_count > 0:
                    image_dict[class_name] = qualifying_image_count

        # Add to result
        if image_dict:  # Only add if there's at least one class with qualifying images
            result[f"{dir_name}.jpg"] = image_dict

# Write to JSON file
output_file = OUTPUTS_ROOT / 'animals' / 'animal_pred_1.json'
with open(output_file, 'w') as f:
    json.dump(result, f, indent=4)

print(f"Data written to {output_file}")
