import argparse
import logging
import os
from glob import glob

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm

from marigold import MarigoldPipeline
from marigold.util.seed_all import seed_all

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )
    parser.add_argument("--checkpoint", type=str, default="Bingxin/Marigold")
    parser.add_argument("--input_rgb_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--denoise_steps", type=int, default=10)
    parser.add_argument("--ensemble_size", type=int, default=10)
    parser.add_argument("--half_precision", action="store_true")
    parser.add_argument("--processing_res", type=int, default=768)
    parser.add_argument("--output_processing_res", action="store_true")
    parser.add_argument("--color_map", type=str, default="Spectral")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--apple_silicon", action="store_true")
    args = parser.parse_args()

    if args.ensemble_size > 15:
        logging.warning("Running with large ensemble size will be slow.")
    if args.apple_silicon and args.batch_size == 0:
        args.batch_size = 1

    if args.seed is None:
        import time

        args.seed = int(time.time())
    seed_all(args.seed)

    output_dir_color = os.path.join(args.output_dir, "depth_colored")
    output_dir_tif = os.path.join(args.output_dir, "depth_bw")
    output_dir_npy = os.path.join(args.output_dir, "depth_npy")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(output_dir_color, exist_ok=True)
    os.makedirs(output_dir_tif, exist_ok=True)
    os.makedirs(output_dir_npy, exist_ok=True)

    if args.apple_silicon:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps:0")
        else:
            device = torch.device("cpu")
            logging.warning("MPS is not available. Running on CPU will be slow.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")

    rgb_filename_list = glob(os.path.join(args.input_rgb_dir, "*"))
    rgb_filename_list = [
        f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    ]
    rgb_filename_list = sorted(rgb_filename_list)
    if not rgb_filename_list:
        raise RuntimeError(f"No image found in '{args.input_rgb_dir}'")

    dtype = torch.float16 if args.half_precision else torch.float32
    pipe = MarigoldPipeline.from_pretrained(args.checkpoint, torch_dtype=dtype)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    pipe = pipe.to(device)

    with torch.no_grad():
        for rgb_path in tqdm(rgb_filename_list, desc="Estimating depth", leave=True):
            try:
                input_image = Image.open(rgb_path)
            except UnidentifiedImageError:
                logging.error("Could not identify image file %s. Skipping...", rgb_path)
                continue

            pipe_out = pipe(
                input_image,
                denoising_steps=args.denoise_steps,
                ensemble_size=args.ensemble_size,
                processing_res=args.processing_res,
                match_input_res=not args.output_processing_res,
                batch_size=args.batch_size,
                color_map=args.color_map,
                show_progress_bar=True,
            )

            depth_pred = pipe_out.depth_np
            depth_colored = pipe_out.depth_colored
            rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
            pred_name_base = rgb_name_base + "_pred"

            np.save(os.path.join(output_dir_npy, f"{pred_name_base}.npy"), depth_pred)
            depth_to_save = (depth_pred * 65535.0).astype(np.uint16)
            Image.fromarray(depth_to_save).save(
                os.path.join(output_dir_tif, f"{pred_name_base}.png"), mode="I;16"
            )
            depth_colored.save(
                os.path.join(output_dir_color, f"{pred_name_base}_colored.png")
            )


if __name__ == "__main__":
    main()
