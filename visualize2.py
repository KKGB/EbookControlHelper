import cv2
import numpy as np
from pathlib import Path

# 경로 설정
LABEL_ROOT = Path("data/labels/train")  # YOLO .txt 경로
IMG_ROOT = Path("data/images/train")    # 이미지 경로
OUT_ROOT = Path("visualization")  # raster 기반 결과 저장 경로

CLASS_NAMES = [
    "right_iris", "left_iris",
    "right_eyelid_open", "right_eyelid_closed",
    "left_eyelid_open", "left_eyelid_closed"
]

COLORS = [
    (255, 64, 64),   # red
    (64, 64, 255),   # blue
    (255, 192, 0),   # yellow
    (255, 128, 0),   # orange
    (0, 255, 128),   # green
    (0, 128, 255),   # sky
]

OUT_ROOT.mkdir(parents=True, exist_ok=True)

def visualize_fillpoly(txt_path: Path):
    name = txt_path.stem
    rel_path = txt_path.relative_to(LABEL_ROOT).with_suffix(".jpg")
    img_path = IMG_ROOT / rel_path
    out_path = OUT_ROOT / rel_path

    if not img_path.exists():
        print(f"[!] Image not found for {txt_path.name}")
        return

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[!] Failed to read {img_path}")
        return

    h, w = img.shape[:2]
    overlay = img.copy()

    with open(txt_path, "r") as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) < 6:
                continue
            cls_id = int(vals[0])
            pts = vals[5:]

            polygon = [(int(pts[i]*w), int(pts[i+1]*h)) for i in range(0, len(pts), 2)]
            color = COLORS[cls_id % len(COLORS)]
            poly_np = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)

            # fillPoly로 마스크처럼 덧씌우기
            cv2.fillPoly(overlay, [poly_np], color)

            # class 텍스트 추가
            cv2.putText(overlay, CLASS_NAMES[cls_id], polygon[0],
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    print(f"✓ Saved: {out_path}")

# ────────────── 예시 일부 실행 ──────────────
all_txt = sorted(LABEL_ROOT.rglob("*.txt"))[10000:10004]
for txt_path in all_txt:
    visualize_fillpoly(txt_path)
