# masc_process.py

import os
import cv2
import numpy as np
from scipy.io import savemat
from glob import glob

def process_masc_images(root_dir, output_dir, crop_size=256):
    """
    Process all .png images in a MASC campaign folder structure,
    detect snowflakes and save cropped images and descriptor .mat files.

    Parameters:
    - root_dir: Root path to campaign folder (with camera subfolders)
    - output_dir: Where to save results (cropped images and .mat files)
    - crop_size: Minimum crop size (if larger than object, will be padded)
    """
    all_pngs = glob(os.path.join(root_dir, "**", "*.png"), recursive=True)
    print(f"Found {len(all_pngs)} PNG images under {root_dir}")

    for img_path in all_pngs:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # --- Improved ROI detection ---
        _, binary = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Skip small contours (noise)
        contours = [c for c in contours if cv2.contourArea(c) > 500]
        if not contours:
            print(f"No significant object found in {img_path}")
            continue

        # Take the largest contour only
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)

        # Add margin around the bounding box
        padding_ratio = 0.5
        pad_w = int(w * padding_ratio)
        pad_h = int(h * padding_ratio)

        xmin = max(0, x - pad_w)
        ymin = max(0, y - pad_h)
        xmax = min(img.shape[1], x + w + pad_w)
        ymax = min(img.shape[0], y + h + pad_h)
        crop = img[ymin:ymax, xmin:xmax]

        # Build output paths mirroring input folder structure
        rel_path = os.path.relpath(img_path, root_dir)
        rel_dir = os.path.dirname(rel_path)
        out_subdir = os.path.join(output_dir, rel_dir)
        os.makedirs(out_subdir, exist_ok=True)

        base = os.path.splitext(os.path.basename(img_path))[0]
        crop_name = f"{base}_roi.png"
        mat_name = f"{base}_roi.mat"

        out_crop_path = os.path.join(out_subdir, crop_name)
        out_mat_path = os.path.join(out_subdir, mat_name)

        cv2.imwrite(out_crop_path, crop)

        # Example descriptors
        descriptors = {
            'area': float(cv2.contourArea(cnt)),
            'perimeter': float(cv2.arcLength(cnt, True)),
            'bbox': [int(x), int(y), int(w), int(h)],
            'padded_bbox': [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)]
        }

        savemat(out_mat_path, descriptors)

        print(f"Processed {img_path} with crop saved to {crop_name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process MASC PNG images and extract snowflake crops + features")
    parser.add_argument("--input", required=True, help="Root input folder (campaign)")
    parser.add_argument("--output", required=True, help="Output folder for processed data")
    parser.add_argument("--crop", type=int, default=256, help="Minimum crop size (used if no bbox) — unused in this version")
    args = parser.parse_args()

    process_masc_images(args.input, args.output, args.crop)
