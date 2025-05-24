import cv2
import numpy as np
from pathlib import Path

# 경로 설정
LABEL_ROOT = Path("data/labels/train")  # YOLO .txt 경로
IMG_ROOT = Path("data/images/train")    # 이미지 경로
OUT_ROOT = Path("visualization")          # 결과 저장 경로

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

def draw_seg_mask(img, poly, color, label):
    poly = np.array(poly, dtype=np.int32).reshape(-1, 2)
    cv2.polylines(img, [poly], isClosed=True, color=color, thickness=2)
    cv2.putText(img, label, tuple(poly[0]),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=color, thickness=1, lineType=cv2.LINE_AA)

def draw_bbox(img, cx, cy, bw, bh, color):
    h, w = img.shape[:2]
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=1, lineType=cv2.LINE_AA)

def visualize(txt_path: Path):
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
    with open(txt_path, "r") as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) < 6:
                continue
            cls_id = int(vals[0])
            cx, cy, bw, bh = vals[1:5]
            pts = vals[5:]

            # draw polygon
            poly = [(pts[i]*w, pts[i+1]*h) for i in range(0, len(pts), 2)]
            color = COLORS[cls_id % len(COLORS)]
            label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            draw_seg_mask(img, poly, color, label)

            # draw bbox
            draw_bbox(img, cx, cy, bw, bh, color)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"✓ Saved: {out_path}")

# ────────────── 최대 10개만 실행 ──────────────
all_txt = sorted(LABEL_ROOT.rglob("*.txt"))[10000:10004]
for txt_path in all_txt:
    visualize(txt_path)
