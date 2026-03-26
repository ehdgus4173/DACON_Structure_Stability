import os
import cv2
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import TRAIN_CSV, DEV_CSV, PROJECT_ROOT, DATASET_DIR

def extract_image_features(img_np):
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    
    mean_brightness = np.mean(img_gray)
    rms_contrast = np.std(img_gray)
    
    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = np.sum(edges > 0) / (img_gray.shape[0] * img_gray.shape[1])
    
    h_mean = np.mean(img_hsv[:, :, 0])
    s_mean = np.mean(img_hsv[:, :, 1])
    v_mean = np.mean(img_hsv[:, :, 2])
    
    return {
        "brightness": mean_brightness,
        "contrast": rms_contrast,
        "edge_density": edge_density,
        "hsv_h_mean": h_mean,
        "hsv_s_mean": s_mean,
        "hsv_v_mean": v_mean
    }

def extract_structure_mask(img_np, view="front"):
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    mask = (sat > 40).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def extract_features_front(img_np):
    H, W = img_np.shape[:2]
    feats = {}
    mask = extract_structure_mask(img_np, "front")
    pts = cv2.findNonZero(mask)

    if pts is None or len(pts) < 20:
        return {k: 0.0 for k in [
            "f_tilt_angle", "f_cx_norm", "f_cy_norm",
            "f_cx_offset", "f_cy_offset", "f_height_ratio",
            "f_width_ratio", "f_aspect_ratio", "f_bbox_area_ratio",
            "f_top_width_ratio", "f_mass_upper_ratio"
        ]}

    rect = cv2.minAreaRect(pts)
    (cx, cy), (rw, rh), angle = rect
    if rw < rh:
        angle = angle + 90
    tilt = abs(angle % 90)
    feats["f_tilt_angle"] = tilt

    M = cv2.moments(mask)
    if M["m00"] > 0:
        gcx = M["m10"] / M["m00"]
        gcy = M["m01"] / M["m00"]
    else:
        gcx, gcy = cx, cy
    feats["f_cx_norm"]   = gcx / W
    feats["f_cy_norm"]   = gcy / H
    feats["f_cx_offset"] = abs(gcx / W - 0.5)
    feats["f_cy_offset"] = gcy / H - 0.5

    x, y, bw, bh = cv2.boundingRect(pts)
    feats["f_height_ratio"]    = bh / H
    feats["f_width_ratio"]     = bw / W
    feats["f_aspect_ratio"]    = bh / (bw + 1e-6)
    feats["f_bbox_area_ratio"] = (bw * bh) / (W * H)

    mid_y = int(gcy)
    top_pts = [p for p in pts.reshape(-1, 2) if p[1] < mid_y]
    bot_pts = [p for p in pts.reshape(-1, 2) if p[1] >= mid_y]
    top_w = (max(p[0] for p in top_pts) - min(p[0] for p in top_pts)) if len(top_pts) > 2 else 0
    bot_w = (max(p[0] for p in bot_pts) - min(p[0] for p in bot_pts)) if len(bot_pts) > 2 else 1
    feats["f_top_width_ratio"] = top_w / (bot_w + 1e-6)

    upper_mass = mask[:mid_y, :].sum()
    lower_mass = mask[mid_y:, :].sum()
    feats["f_mass_upper_ratio"] = upper_mass / (upper_mass + lower_mass + 1e-6)

    return feats

def extract_features_top(img_np):
    H, W = img_np.shape[:2]
    feats = {}
    mask = extract_structure_mask(img_np, "top")
    pts = cv2.findNonZero(mask)

    if pts is None or len(pts) < 10:
        return {k: 0.0 for k in [
            "t_cx_offset", "t_cy_offset", "t_footprint_area",
            "t_aspect_ratio", "t_footprint_tilt",
            "t_left_mass_ratio", "t_compactness"
        ]}

    M = cv2.moments(mask)
    gcx = M["m10"] / M["m00"] if M["m00"] > 0 else W / 2
    gcy = M["m01"] / M["m00"] if M["m00"] > 0 else H / 2
    feats["t_cx_offset"] = abs(gcx / W - 0.5)
    feats["t_cy_offset"] = abs(gcy / H - 0.5)

    area = mask.sum() / 255
    feats["t_footprint_area"] = area / (W * H)

    rect = cv2.minAreaRect(pts)
    (_, _), (rw, rh), angle = rect
    long_side  = max(rw, rh)
    short_side = min(rw, rh) + 1e-6
    feats["t_aspect_ratio"]    = long_side / short_side
    feats["t_footprint_tilt"]  = abs(angle % 90)

    left_mass  = mask[:, :W//2].sum()
    right_mass = mask[:, W//2:].sum()
    total = left_mass + right_mass + 1e-6
    feats["t_left_mass_ratio"] = abs(left_mass / total - 0.5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest, True) + 1e-6
        contour_area = cv2.contourArea(largest) + 1e-6
        feats["t_compactness"] = 4 * np.pi * contour_area / (perimeter ** 2)
    else:
        feats["t_compactness"] = 0.0

    return feats

def extract_all_features(sample_dir: Path):
    front_path = sample_dir / "front.png"
    top_path = sample_dir / "top.png"
    
    if not front_path.exists() or not top_path.exists():
        return None
        
    front_img = cv2.cvtColor(cv2.imread(str(front_path)), cv2.COLOR_BGR2RGB)
    top_img = cv2.cvtColor(cv2.imread(str(top_path)), cv2.COLOR_BGR2RGB)
    
    feats = {}
    
    p_front = extract_image_features(front_img)
    p_top = extract_image_features(top_img)
    for k, v in p_front.items():
        feats[f"front_{k}"] = v
    for k, v in p_top.items():
        feats[f"top_{k}"] = v
        
    feats.update(extract_features_front(front_img))
    feats.update(extract_features_top(top_img))
    
    return feats

def main(args=None):
    parser = argparse.ArgumentParser()
    parsed_args = parser.parse_args(args if args is not None else [])
    
    test_csv_path = PROJECT_ROOT.parent / "EDA" / "codebase" / "example" / "sample_submission.csv"
    dfs = {
        "train": pd.read_csv(TRAIN_CSV),
        "dev": pd.read_csv(DEV_CSV)
    }
    if test_csv_path.exists():
        dfs["test"] = pd.read_csv(test_csv_path)

    all_features = []
    
    print("Extracting pixel-level and physical features directly from images...")
    for split_name, df in dfs.items():
        base_dir = DATASET_DIR / split_name
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
            sample_id = str(row['id'])
            sample_dir = base_dir / sample_id
            
            feats = extract_all_features(sample_dir)
            if feats is not None:
                feats["id"] = sample_id
                all_features.append(feats)

    pixel_df = pd.DataFrame(all_features)
    print(f"Extraction complete! Row count: {len(pixel_df)}")
    
    eda_base = PROJECT_ROOT.parent / "EDA" / "codebase" / "0324_제공데이터분석"
    video_csv = eda_base / "04_mp4_ana" / "outputs" / "video_motion_features.csv"
    
    combined_df = pixel_df.copy()
    
    if video_csv.exists():
        v_df = pd.read_csv(video_csv).drop(columns=['label', 'label_bin'], errors='ignore')
        combined_df = combined_df.merge(v_df, on='id', how='left')
        print(f"Merged video motion features. Columns: {len(combined_df.columns)}")
        
    output_dir = PROJECT_ROOT / "features"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "combined_features_v2.csv"
    
    combined_df.fillna(0, inplace=True)
    combined_df.to_csv(out_path, index=False)
    
    print(f"✅ Final combined feature shape: {combined_df.shape}")
    print(f"✅ Saved to: {out_path}")

if __name__ == "__main__":
    main()
