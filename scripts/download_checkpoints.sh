#!/bin/bash
# Download all required model checkpoints for OmniCount
# Usage: bash scripts/download_checkpoints.sh [output_dir]
# Default output directory: ./checkpoints

set -e

OUTPUT_DIR="${1:-./checkpoints}"
mkdir -p "$OUTPUT_DIR"

echo "Downloading checkpoints to $OUTPUT_DIR..."

# SAM (Segment Anything Model) checkpoints
SAM_VIT_B="$OUTPUT_DIR/sam_vit_b_01ec64.pth"
SAM_VIT_H="$OUTPUT_DIR/sam_vit_h_4b8939.pth"

if [ ! -f "$SAM_VIT_B" ]; then
    echo "Downloading SAM ViT-B checkpoint..."
    wget -O "$SAM_VIT_B" "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
else
    echo "SAM ViT-B checkpoint already exists."
fi

if [ ! -f "$SAM_VIT_H" ]; then
    echo "Downloading SAM ViT-H checkpoint..."
    wget -O "$SAM_VIT_H" "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
else
    echo "SAM ViT-H checkpoint already exists."
fi

# SAN (Segment Any Class Network) checkpoints
SAN_VIT_B="$OUTPUT_DIR/san_vit_b_16.pth"
SAN_VIT_L="$OUTPUT_DIR/san_vit_large_14.pth"

if [ ! -f "$SAN_VIT_B" ]; then
    echo "Downloading SAN ViT-B/16 checkpoint..."
    # Replace with actual URL if available
    echo "Note: Please download SAN ViT-B/16 checkpoint manually and place it at: $SAN_VIT_B"
    echo "      Refer to https://github.com/MendelXu/SAN for checkpoint download instructions."
fi

if [ ! -f "$SAN_VIT_L" ]; then
    echo "Downloading SAN ViT-L/14 checkpoint..."
    # Replace with actual URL if available
    echo "Note: Please download SAN ViT-L/14 checkpoint manually and place it at: $SAN_VIT_L"
    echo "      Refer to https://github.com/MendelXu/SAN for checkpoint download instructions."
fi

# GroundingDINO checkpoints
GDINO_SWINT="$OUTPUT_DIR/groundingdino_swint_ogc.pth"
GDINO_SWINB="$OUTPUT_DIR/groundingdino_swinb_cogcoor.pth"

if [ ! -f "$GDINO_SWINT" ]; then
    echo "Downloading GroundingDINO SwinT checkpoint..."
    wget -O "$GDINO_SWINT" "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
else
    echo "GroundingDINO SwinT checkpoint already exists."
fi

if [ ! -f "$GDINO_SWINB" ]; then
    echo "Downloading GroundingDINO SwinB checkpoint..."
    wget -O "$GDINO_SWINB" "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
else 
    echo "GroundingDINO SwinB checkpoint already exists."
fi

echo ""
echo "=============================="
echo "Checkpoint download complete!"
echo "=============================="
echo "Checkpoints saved to: $OUTPUT_DIR"
echo ""
echo "Note: CLIPSeg model weights (CIDAS/clipseg-rd64-refined) are downloaded"
echo "      automatically via HuggingFace on first use."
