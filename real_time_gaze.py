from collections import deque, Counter
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time

# 서버에서 되는지 확인 안해봄. 로컬에서만 돌려봄.
# PYTORCH_ENABLE_MPS_FALLBACK=1 python real_time_gaze.py
# mac M1/M2 GPU 사용시 PYTORCH_ENABLE_MPS_FALLBACK=1 해야 돌아감
# ───────────── 설정
MODEL_PATH  = "best_s.pt"
CAM_ID      = 0
IMG_SIZE    = 640
CONF_THRES  = 0.25
IOU_THRES   = 0.3
FPS         = 10 # 10 FPS (환경마다 다를 것이므로, 밑에 FPS 출력하는 부분 주석 풀고 확인해보시길)  (print(f"FPS: {fps:.2f}"))
WINDOW_SEC  = 3 # 3초 동안의 gaze 방향을 확인
TOLERANCE   = 0.1  # 10% 허용 (3초 중 10%는 다른 방향이여도 현재 dominant 방향으로 인정)

gaze_directions = {0: "Right", 1: "Left", 2: "Center"}
required_frames = int(FPS * WINDOW_SEC)
min_agreement   = int(required_frames * (1 - TOLERANCE))
DIRECTION_BUFFER = deque(maxlen=required_frames)

# ───────────── 색상
COLORS = [
    (255,  64,  64),  # right_iris
    ( 64,  64, 255),  # left_iris
    (255, 192,   0),  # right_eyelid
    (  0, 255, 128),  # left_eyelid
]

# ───────────── 함수
def get_center(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.mean()), int(ys.mean())

def put_status_bar(frame, left_eye, right_eye):
    h, w = frame.shape[:2]
    bar_h = 60
    cv2.rectangle(frame, (0, h-bar_h), (w, h), (0, 0, 0), -1)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
    cv2.putText(frame, f"Left  : {left_eye}",  (20,  h-20),
                font, scale, (255,255,255), thick, cv2.LINE_AA)
    cv2.putText(frame, f"Right : {right_eye}", (w//2+20, h-20),
                font, scale, (255,255,255), thick, cv2.LINE_AA)
    return frame

def draw_direction(frame, direction):
    cv2.putText(frame, f"Gaze: {direction}", (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4, cv2.LINE_AA)
    return frame

# ───────────── 모델 준비
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)
model = YOLO(MODEL_PATH)
model.fuse()

# ───────────── 웹캠 루프
cap = cv2.VideoCapture(CAM_ID)
if not cap.isOpened():
    raise RuntimeError("Webcam 열기 실패")

print("🔵 Press [q] to quit")

confirmed_gaze = None
prev_time = time.time()
fps = 0

with torch.no_grad():
    while True:
        current_time = time.time()
        elapsed = current_time - prev_time
        fps = 1 / elapsed if elapsed > 0 else 0
        prev_time = current_time

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # YOLO 추론
        res = model(frame, imgsz=IMG_SIZE,
                    conf=CONF_THRES, iou=IOU_THRES,
                    device=device, verbose=False)[0]

        left_eye_state, right_eye_state = "Closed", "Closed"
        iris_mask = lid_mask = None

        if res.masks is not None:
            masks = (res.masks.data > 0.5).cpu().numpy()
            classes = res.boxes.cls.int().cpu().tolist()

            has_left_iris  = any(cls == 0 for cls in classes)
            has_right_iris = any(cls == 1 for cls in classes)
            has_left_lid   = any(cls == 2 for cls in classes)
            has_right_lid  = any(cls == 3 for cls in classes)

            if has_left_lid:
                left_eye_state  = "Open" if has_left_iris else "Closed"
            if has_right_lid:
                right_eye_state = "Open" if has_right_iris else "Closed"

            # 중심점 계산용 마스크 분류
            for mask, cls in zip(masks, classes):
                if cls == 1: iris_mask = mask  # right_iris
                if cls == 3: lid_mask  = mask  # right_eyelid

            # gaze 추적
            if iris_mask is not None and lid_mask is not None:
                iris_c = get_center(iris_mask)
                lid_c  = get_center(lid_mask)
                if iris_c and lid_c:
                    dx = iris_c[0] - lid_c[0]
                    if dx > 5:
                        current_gaze = 0  # Right
                    elif dx < -5:
                        current_gaze = 1  # Left
                    else:
                        current_gaze = 2  # Center
                    print(gaze_directions[current_gaze])
                    DIRECTION_BUFFER.append(current_gaze)

                    # 3초간 거의 같은 방향이면 확정
                    if len(DIRECTION_BUFFER) == required_frames:
                        counts = Counter(DIRECTION_BUFFER)
                        most_common, count = counts.most_common(1)[0]
                        if count >= min_agreement and most_common != confirmed_gaze:
                            confirmed_gaze = most_common
                            print("👁 Gaze:", gaze_directions[confirmed_gaze])
                        DIRECTION_BUFFER.clear()

            # ─── 마스크 오버레이 시각화
            overlay = frame.copy()
            for mask, cls in zip(masks, classes):
                col = COLORS[cls % len(COLORS)]
                mask = cv2.resize(mask.astype(np.uint8), (w, h),
                                  interpolation=cv2.INTER_NEAREST)
                for c in range(3):
                    overlay[:,:,c] = np.where(mask,
                        0.4*col[c] + 0.6*overlay[:,:,c], overlay[:,:,c])
            frame = overlay
        # print(f"FPS: {fps:.2f}")
        # ─── 출력
        if confirmed_gaze is not None:
            frame = draw_direction(frame, gaze_directions[confirmed_gaze])
        frame = put_status_bar(frame, left_eye_state, right_eye_state)

        cv2.imshow("Eye Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
