#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGE_DIR="${IMAGE_DIR:-$REPO_ROOT/../../Datasets/UCF_CC_50/images}"
VOCAB_FILE="$REPO_ROOT/outputs_sota/ucf-50/ucf_classes.txt"
OUTPUT_BASE_DIR="$REPO_ROOT/outputs_sota/ucf-50/binary_masks"
CONFIG_FILE="$REPO_ROOT/external/SAN/configs/san_clip_vit_large_res4_coco.yaml"
MODEL_PATH="$REPO_ROOT/external/SAN/models/san_vit_large_14.pth"
PREDICT_SCRIPT="$REPO_ROOT/external/SAN/predict.py"

if [ ! -d "$IMAGE_DIR" ]; then
    echo "Image directory not found: $IMAGE_DIR"
    echo "Set IMAGE_DIR before running this script."
    exit 1
fi

if [ ! -f "$VOCAB_FILE" ]; then
    echo "Vocabulary file not found: $VOCAB_FILE"
    exit 1
fi

while IFS=$'\t' read -r image_name vocab; do
    img_path="${IMAGE_DIR}/${image_name}"
    output_dir="${OUTPUT_BASE_DIR}/${image_name%.*}"
    mkdir -p "$output_dir"

    IFS=',' read -ra vocab_array <<< "$vocab"
    for class_name in "${vocab_array[@]}"; do
        output_file="${output_dir}/output_mask_${class_name}.jpg"
        echo "Processing $img_path with class $class_name..."
        python "$PREDICT_SCRIPT" \
            --config-file "$CONFIG_FILE" \
            --model-path "$MODEL_PATH" \
            --img-path "$img_path" \
            --vocab "$class_name" \
            --output-file "$output_file"
    done
done < "$VOCAB_FILE"

echo "All images processed."
