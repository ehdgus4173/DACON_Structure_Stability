"""
Physics Features Extraction Module

학술 근거:
🟢 Lerer et al.(2016) PhysNet, ECCV
🟢 Battaglia et al.(2016) Interaction Networks, NeurIPS
🟢 Otsu(1979) — OTSU 이진화
🟡 CoM-BoS Framework
"""

import cv2
import numpy as np

def _preprocess_image(img_np: np.ndarray) -> np.ndarray:
    """
    HSV 채도 기반 구조물 검출 — 적응형 임계값
    고정 임계값 80이 저채도 블록 샘플에서 실패하는 문제 해결
    🔴 fallback 순서는 휴리스틱 — 04_experiments에서 검증 필요
    """
    if img_np is None or img_np.size == 0:
        return None

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    h, w = img_np.shape[:2]
    min_area = h * w * 0.005

    # 적응형 임계값: 80 → 50 → 30 순서로 fallback
    for thresh in [80, 50, 30]:
        mask = (s > thresh).astype(np.uint8) * 255

        # morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)

        if cv2.contourArea(cnt) >= min_area:
            return cnt  # 유효한 윤곽선 찾으면 즉시 반환

    return None  # 모든 임계값에서 실패

def extract_structure_tilt_angle(front_img: np.ndarray) -> float:
    # 🟢 Lerer et al.(2016) PhysNet — 기울기가 붕괴 예측 핵심 신호
    cnt = _preprocess_image(front_img)
    if cnt is None:
        return 0.0
        
    rect = cv2.minAreaRect(cnt)
    _, (w, h), angle = rect
    
    if w < h:
        tilt = abs(angle)
    else:
        tilt = abs(90 - angle)
        
    return float(min(tilt, 90.0))

def extract_height_to_base_ratio(front_img: np.ndarray) -> float:
    # 🟢 정역학 기본 원리 — 세장비 클수록 전도 모멘트 증가
    cnt = _preprocess_image(front_img)
    if cnt is None:
        return 0.0
        
    x, y, w, h = cv2.boundingRect(cnt)
    if w == 0:
        return 0.0
        
    return float(h / w)

def extract_footprint_compactness(top_img: np.ndarray) -> float:
    # 🟡 CoM-BoS Framework — 기저면 형태가 방향 취약성 결정
    cnt = _preprocess_image(top_img)
    if cnt is None:
        return 0.0
        
    rect = cv2.minAreaRect(cnt)
    w, h = rect[1]
    if max(w, h) == 0:
        return 0.0
        
    return float(min(w, h) / max(w, h))

def extract_bounding_box_aspect_skew(front_img: np.ndarray) -> float:
    # 🟢 Battaglia et al.(2016) — 무게중심 편차가 불안정성 예측에 기여
    cnt = _preprocess_image(front_img)
    if cnt is None:
        return 0.0
        
    x, y, w, h = cv2.boundingRect(cnt)
    if w == 0:
        return 0.0
        
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return 0.0
        
    cx = M["m10"] / M["m00"]
    box_center_x = x + w / 2.0
    skew = (cx - box_center_x) / w
    
    return float(np.clip(skew, -1.0, 1.0))

def extract_com_horizontal_offset(front_img: np.ndarray) -> float:
    # 🟢 Battaglia et al.(2016) — CoM-BoS 이탈이 불안정성 직접 지표
    cnt = _preprocess_image(front_img)
    if cnt is None:
        return 0.0
        
    img_w = front_img.shape[1]
    if img_w == 0:
        return 0.0
        
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return 0.0
        
    cx = M["m10"] / M["m00"]
    img_center_x = img_w / 2.0
    offset = (cx - img_center_x) / img_w
    
    return float(np.clip(offset, -1.0, 1.0))

def extract_top_com_deviation(top_img: np.ndarray) -> float:
    # 🟡 CoM-BoS Framework — Top view 편차가 붕괴 방향 예측에 활용
    cnt = _preprocess_image(top_img)
    if cnt is None:
        return 0.0
        
    h_img, w_img = top_img.shape[:2]
    diag = np.sqrt(w_img**2 + h_img**2)
    if diag == 0:
        return 0.0
        
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return 0.0
        
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    
    img_cx, img_cy = w_img / 2.0, h_img / 2.0
    dev = np.sqrt((cx - img_cx)**2 + (cy - img_cy)**2) / diag
    
    return float(dev)

def extract_physics_features(front_img: np.ndarray, top_img: np.ndarray) -> np.ndarray:
    # 🔴 fallback=0.0 휴리스틱 — 추출 실패 샘플 영향도 04_experiments에서 Ablation 필요
    features = [0.0] * 6
    
    funcs = [
        (extract_structure_tilt_angle, front_img, 0),
        (extract_height_to_base_ratio, front_img, 1),
        (extract_footprint_compactness, top_img, 2),
        (extract_bounding_box_aspect_skew, front_img, 3),
        (extract_com_horizontal_offset, front_img, 4),
        (extract_top_com_deviation, top_img, 5)
    ]
    
    for func, img, idx in funcs:
        try:
            val = func(img)
            features[idx] = float(val)
        except Exception:
            features[idx] = 0.0
            
    feat_arr = np.array(features, dtype=np.float32)
    feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)
    
    return feat_arr

# SHAP 분석 노트북에서 피처 이름 레이블로 사용
# extract_physics_features 반환 배열의 인덱스와 순서가 일치해야 함
PHYSICS_FEATURE_NAMES = [
    "Structure_Tilt_Angle",      # index 0 — front
    "Height_to_Base_Ratio",      # index 1 — front
    "Footprint_Compactness",     # index 2 — top
    "Bounding_Box_Aspect_Skew",  # index 3 — front
    "CoM_Horizontal_Offset",     # index 4 — front
    "Top_CoM_Deviation"          # index 5 — top
]
