import cv2
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import TRAIN_CSV, DEV_CSV, PROJECT_ROOT, DATASET_DIR


# ================================================================
# 이미지 전반 통계 피처
# ================================================================

def extract_image_features(img_np):
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    img_hsv  = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    edges = cv2.Canny(img_gray, 50, 150)

    return {
        "brightness":   np.mean(img_gray),
        "contrast":     np.std(img_gray),
        "edge_density": np.sum(edges > 0) / img_gray.size,
        "hsv_h_mean":   np.mean(img_hsv[:, :, 0]),
        "hsv_s_mean":   np.mean(img_hsv[:, :, 1]),
        "hsv_v_mean":   np.mean(img_hsv[:, :, 2]),
    }


# ================================================================
# 구조물 마스크 추출
# ================================================================

def extract_structure_mask(img_np):
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    mask = (hsv[:, :, 1] > 40).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return mask


# ================================================================
# Hough 라인 기반 격자 검출 유틸리티
# ================================================================

def _detect_grid_lines(gray: np.ndarray):
    """
    Hough 변환으로 격자선을 검출하고 수평/수직 두 클러스터로 분리한다.

    반환:
        h_lines : 수평 방향 선 목록  [(x1,y1,x2,y2), ...]
        v_lines : 수직 방향 선 목록  [(x1,y1,x2,y2), ...]
        검출 실패 시 (None, None)
    """
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)

    H, W = gray.shape
    min_len = min(H, W) // 10
    lines = cv2.HoughLinesP(
        edges,
        rho=1, theta=np.pi / 180,
        threshold=40,
        minLineLength=min_len,
        maxLineGap=min_len // 2,
    )

    if lines is None or len(lines) < 4:
        return None, None

    lines = lines.reshape(-1, 4)

    angles = np.degrees(np.arctan2(
        lines[:, 3] - lines[:, 1],
        lines[:, 2] - lines[:, 0],
    ))
    angles = ((angles + 90) % 180) - 90

    h_mask = np.abs(angles) < 45
    v_mask = ~h_mask

    h_lines = lines[h_mask].tolist() if h_mask.sum() >= 2 else None
    v_lines = lines[v_mask].tolist() if v_mask.sum() >= 2 else None

    return h_lines, v_lines


def _lines_to_angles(lines) -> np.ndarray:
    """선 목록에서 각도(도) 배열을 반환."""
    arr = np.array(lines)
    return np.degrees(np.arctan2(arr[:, 3] - arr[:, 1], arr[:, 2] - arr[:, 0]))


def _vanishing_point(lines):
    """선 목록에서 최소제곱으로 소실점을 추정한다."""
    if lines is None or len(lines) < 2:
        return None

    arr = np.array(lines, dtype=np.float64)
    x1, y1, x2, y2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    a = y2 - y1
    b = -(x2 - x1)
    c = (x2 - x1) * y1 - (y2 - y1) * x1

    norms = np.sqrt(a ** 2 + b ** 2) + 1e-9
    A = np.stack([a / norms, b / norms], axis=1)
    B = -c / norms

    result, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return result


# ================================================================
# Hough 기반 격자 시점 피처 추출
# ================================================================

def extract_grid_features(img_np: np.ndarray, view: str = 'top') -> dict:
    """
    Hough 라인 변환으로 격자선을 검출하고 카메라 시점 피처를 반환한다.

    반환 피처 (prefix = '{view}_grid'):
        _detected          : 수평+수직 격자선 검출 성공 (0 or 1)
        _tilt_angle        : 수평선 클러스터의 평균 기울기(도). 0 = 완전 수평
        _perspective_ratio : 수평선 간격의 원거리/근거리 비.
                             1.0 = 정사영, < 1.0 = 원근 압축

    검출 실패 시 세 피처 모두 0.0으로 fallback.
    """
    prefix = f"{view}_grid"
    fallback = {
        f"{prefix}_detected":          0.0,
        f"{prefix}_tilt_angle":        0.0,
        f"{prefix}_perspective_ratio": 0.0,
    }

    gray    = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h_lines, v_lines = _detect_grid_lines(gray)

    if h_lines is None:
        return fallback

    h_angles = _lines_to_angles(h_lines)
    tilt_angle = float(np.mean(np.abs(h_angles)))

    arr = np.array(h_lines)
    y_centers = (arr[:, 1] + arr[:, 3]) / 2.0
    y_sorted  = np.sort(y_centers)

    if len(y_sorted) >= 4:
        n = len(y_sorted)
        top_gap  = float(np.mean(np.diff(y_sorted[:n // 2])))
        bot_gap  = float(np.mean(np.diff(y_sorted[n // 2:])))
        perspective_ratio = top_gap / (bot_gap + 1e-6)
    else:
        perspective_ratio = 0.0

    return {
        f"{prefix}_detected":          1.0,
        f"{prefix}_tilt_angle":        tilt_angle,
        f"{prefix}_perspective_ratio": perspective_ratio,
    }


# ================================================================
# Hough 기반 호모그래피 보정
# ================================================================

def rectify_by_grid(img_np: np.ndarray) -> np.ndarray:
    """
    Hough 라인으로 격자의 소실점을 추정하고 roll 보정을 적용한다.
    보정 불가 시 원본 이미지 반환 (fallback).
    dataset.py _load_image() 에서 호출.
    """
    H, W   = img_np.shape[:2]
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h_lines, v_lines = _detect_grid_lines(gray)

    if h_lines is None or v_lines is None:
        return img_np

    vp_h = _vanishing_point(h_lines)
    vp_v = _vanishing_point(v_lines)

    if vp_h is None or vp_v is None:
        return img_np

    cx, cy = W / 2.0, H / 2.0
    vp_h_n = vp_h - np.array([cx, cy])
    vp_v_n = vp_v - np.array([cx, cy])

    if (abs(vp_h_n[0]) < W * 0.5) and (abs(vp_h_n[1]) < H * 0.5):
        return img_np
    if (abs(vp_v_n[0]) < W * 0.5) and (abs(vp_v_n[1]) < H * 0.5):
        return img_np

    roll_rad = np.arctan2(vp_v_n[0], max(abs(vp_v_n[1]), 1e-6))

    if abs(roll_rad) < np.radians(1):
        return img_np

    angle_deg = np.degrees(roll_rad)
    M_rot = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rectified = cv2.warpAffine(img_np, M_rot, (W, H),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    return rectified


# ================================================================
# front 뷰 물리 피처
# ================================================================

def extract_features_front(img_np):
    H, W = img_np.shape[:2]
    mask = extract_structure_mask(img_np)
    pts  = cv2.findNonZero(mask)

    if pts is None or len(pts) < 20:
        return {k: 0.0 for k in [
            "f_tilt_angle", "f_cx_norm", "f_cy_norm",
            "f_cx_offset", "f_cy_offset", "f_cy_ratio",
            "f_height_ratio", "f_width_ratio", "f_aspect_ratio",
            "f_bbox_area_ratio", "f_top_width_ratio", "f_mass_upper_ratio",
        ]}

    (cx, cy), (rw, rh), angle = cv2.minAreaRect(pts)
    if rw < rh:
        angle += 90
    feats = {"f_tilt_angle": abs(angle % 90)}

    M = cv2.moments(mask)
    gcx = M["m10"] / M["m00"] if M["m00"] > 0 else cx
    gcy = M["m01"] / M["m00"] if M["m00"] > 0 else cy
    feats["f_cx_norm"]   = gcx / W
    feats["f_cy_norm"]   = gcy / H
    feats["f_cx_offset"] = abs(gcx / W - 0.5)
    feats["f_cy_offset"] = gcy / H - 0.5
    feats["f_cy_ratio"]  = gcy / H          # 수직 무게중심 위치 (0=상단, 1=하단)

    x, y, bw, bh = cv2.boundingRect(pts)
    feats["f_height_ratio"]    = bh / H
    feats["f_width_ratio"]     = bw / W
    feats["f_aspect_ratio"]    = bh / (bw + 1e-6)
    feats["f_bbox_area_ratio"] = (bw * bh) / (W * H)

    mid_y  = int(gcy)
    pts_2d = pts.reshape(-1, 2)
    top_xs = pts_2d[pts_2d[:, 1] < mid_y, 0]
    bot_xs = pts_2d[pts_2d[:, 1] >= mid_y, 0]
    top_w  = (top_xs.max() - top_xs.min()) if len(top_xs) > 2 else 0
    bot_w  = (bot_xs.max() - bot_xs.min()) if len(bot_xs) > 2 else 1
    feats["f_top_width_ratio"] = top_w / (bot_w + 1e-6)

    upper = mask[:mid_y, :].sum()
    lower = mask[mid_y:, :].sum()
    feats["f_mass_upper_ratio"] = upper / (upper + lower + 1e-6)

    return feats


# ================================================================
# top 뷰 물리 피처
# ================================================================

def extract_features_top(img_np):
    H, W = img_np.shape[:2]
    mask = extract_structure_mask(img_np)
    pts  = cv2.findNonZero(mask)

    if pts is None or len(pts) < 10:
        return {k: 0.0 for k in [
            "t_cx_offset", "t_cy_offset", "t_footprint_area",
            "t_aspect_ratio", "t_footprint_tilt",
            "t_left_mass_ratio", "t_frontback_mass_ratio",
            "t_compactness",
            "t_pa_cx_offset", "t_pa_cy_offset",
        ]}

    M   = cv2.moments(mask)
    gcx = M["m10"] / M["m00"] if M["m00"] > 0 else W / 2
    gcy = M["m01"] / M["m00"] if M["m00"] > 0 else H / 2
    feats = {
        "t_cx_offset":      abs(gcx / W - 0.5),
        "t_cy_offset":      abs(gcy / H - 0.5),
        "t_footprint_area": (mask.sum() / 255) / (W * H),
    }

    (_, _), (rw, rh), angle = cv2.minAreaRect(pts)
    feats["t_aspect_ratio"]   = max(rw, rh) / (min(rw, rh) + 1e-6)
    feats["t_footprint_tilt"] = abs(angle % 90)

    # ── 좌우 / 앞뒤 질량 불균형 ──────────────────────────────────
    left       = mask[:, :W//2].sum()
    right      = mask[:, W//2:].sum()
    front_half = mask[:H//2, :].sum()   # 이미지 위 = 카메라 기준 먼 쪽
    back_half  = mask[H//2:, :].sum()   # 이미지 아래 = 카메라 기준 가까운 쪽
    feats["t_left_mass_ratio"]      = abs(left / (left + right + 1e-6) - 0.5)
    feats["t_frontback_mass_ratio"] = abs(front_half / (front_half + back_half + 1e-6) - 0.5)

    # ── Principal Axis 보정 offset ────────────────────────────────
    # minAreaRect angle로 마스크 회전 → 카메라 틸트 제거 후 편심 재계산
    rot_angle = float(angle % 90)
    M_rot     = cv2.getRotationMatrix2D((gcx, gcy), rot_angle, 1.0)
    mask_rot  = cv2.warpAffine(mask, M_rot, (W, H), flags=cv2.INTER_NEAREST)
    M2 = cv2.moments(mask_rot)
    if M2["m00"] > 0:
        pa_cx = M2["m10"] / M2["m00"]
        pa_cy = M2["m01"] / M2["m00"]
        feats["t_pa_cx_offset"] = abs(pa_cx / W - 0.5)
        feats["t_pa_cy_offset"] = abs(pa_cy / H - 0.5)
    else:
        feats["t_pa_cx_offset"] = feats["t_cx_offset"]
        feats["t_pa_cy_offset"] = feats["t_cy_offset"]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest   = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest, True) + 1e-6
        area      = cv2.contourArea(largest) + 1e-6
        feats["t_compactness"] = 4 * np.pi * area / (perimeter ** 2)
    else:
        feats["t_compactness"] = 0.0

    return feats


# ================================================================
# 샘플 단위 통합 추출
# ================================================================

def extract_all_features(sample_dir: Path):
    front_path = sample_dir / "front.png"
    top_path   = sample_dir / "top.png"

    if not front_path.exists() or not top_path.exists():
        return None

    front_img = cv2.cvtColor(cv2.imread(str(front_path)), cv2.COLOR_BGR2RGB)
    top_img   = cv2.cvtColor(cv2.imread(str(top_path)),   cv2.COLOR_BGR2RGB)

    feats = {}
    for k, v in extract_image_features(front_img).items():
        feats[f"front_{k}"] = v
    for k, v in extract_image_features(top_img).items():
        feats[f"top_{k}"] = v

    feats.update(extract_features_front(front_img))
    feats.update(extract_features_top(top_img))

    feats.update(extract_grid_features(front_img, view='front'))
    feats.update(extract_grid_features(top_img,   view='top'))

    return feats


# ================================================================
# 메인 실행
# ================================================================

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.parse_args(args if args is not None else [])

    test_csv_path = PROJECT_ROOT.parent / "EDA" / "codebase" / "example" / "sample_submission.csv"
    dfs = {
        "train": pd.read_csv(TRAIN_CSV),
        "dev":   pd.read_csv(DEV_CSV),
    }
    if test_csv_path.exists():
        dfs["test"] = pd.read_csv(test_csv_path)

    all_features = []
    print("Extracting pixel + physics + grid features from images (train/dev/test)...")
    for split_name, df in dfs.items():
        base_dir = DATASET_DIR / split_name
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
            sample_id = str(row['id'])
            feats = extract_all_features(base_dir / sample_id)
            if feats is not None:
                feats["id"] = sample_id
                all_features.append(feats)

    combined_df = pd.DataFrame(all_features)
    print(f"Extraction complete: {len(combined_df)} samples")

    for view in ['front', 'top']:
        col = f"{view}_grid_detected"
        if col in combined_df.columns:
            rate = combined_df[col].mean() * 100
            print(f"  {col}: {rate:.1f}% detection rate")

    video_csv = (PROJECT_ROOT.parent / "EDA" / "codebase"
                 / "0324_제공데이터분석" / "04_mp4_ana" / "outputs"
                 / "video_motion_features.csv")
    if video_csv.exists():
        v_df = pd.read_csv(video_csv).drop(columns=['label', 'label_bin'], errors='ignore')
        combined_df = combined_df.merge(v_df, on='id', how='left')
        print(f"Merged video motion features → {len(combined_df.columns)} columns")

    output_dir = PROJECT_ROOT / "features"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "combined_features_v2.csv"

    combined_df.fillna(0, inplace=True)
    combined_df.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path}  shape={combined_df.shape}")


if __name__ == "__main__":
    main()
