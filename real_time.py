import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ─────────────────────────────
# 1. 설정
MODEL_PATH  = "best_s.pt"   # 4-클래스 학습 가중치
CAM_ID      = 0
IMG_SIZE    = 640
CONF_THRES  = 0.25
IOU_THRES   = 0.3
# 클래스 4개 색상(BGR)
COLORS = [
    (255,  64,  64),  # right_iris
    ( 64,  64, 255),  # left_iris
    (255, 192,   0),  # right_eyelid
    (  0, 255, 128),  # left_eyelid
]

# ─────────────────────────────
# 2. 모델 불러오기
device = "mps" if torch.backends.mps.is_available() else "cpu" # MAC 용 gpu device (뭔가를 설치해야하는데 기억안남)
print('device: ',device)
model = YOLO(MODEL_PATH)  # 모델 객체
model.fuse() 

# ─────────────────────────────
def put_status_bar(frame, left_eye, right_eye):
    """하단에 눈 상태(열림/닫힘) 출력"""
    h, w = frame.shape[:2]
    bar_h = 60
    cv2.rectangle(frame, (0, h-bar_h), (w, h), (0, 0, 0), -1)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
    cv2.putText(frame, f"Left  : {left_eye}",  (20,  h-20),
                font, scale, (255,255,255), thick, cv2.LINE_AA)
    cv2.putText(frame, f"Right : {right_eye}", (w//2+20, h-20),
                font, scale, (255,255,255), thick, cv2.LINE_AA)
    return frame

# ─────────────────────────────
# 3. 실시간 루프
cap = cv2.VideoCapture(CAM_ID)
if not cap.isOpened():
    raise RuntimeError("Webcam 열기 실패")

print("🔵 Press [q] to quit")

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)        # 거울 모드
        h, w  = frame.shape[:2]

        # YOLO 추론
        res = model(frame, imgsz=IMG_SIZE,
                    conf=CONF_THRES, iou=IOU_THRES,
                    device=device, verbose=False)[0]

        # 마스크 & 클래스 추출
        left_eye_state, right_eye_state = "Closed", "Closed"
        if res.masks is not None:
            masks   = (res.masks.data > 0.5).cpu().numpy()  # (n,H,W) bool
            classes = res.boxes.cls.int().cpu().tolist()

            # iris 존재 여부 확인
            has_left_iris  = any(cls == 0 for cls in classes) # 1: left_iris -> 원래는 이거지만 좌우 반전해서 보여줘야 하므로 반대로 설정
            has_right_iris = any(cls == 1 for cls in classes) # 0: right_iris

            # eyelid 검출 여부 (원하면 제외해도 OK)
            has_left_lid   = any(cls == 2 for cls in classes) # 3: left_eyelid
            has_right_lid  = any(cls == 3 for cls in classes) # 2: right_eyelid

            # 상태 판정
            if has_left_lid:
                left_eye_state  = "Open" if has_left_iris  else "Closed"
            if has_right_lid:
                right_eye_state = "Open" if has_right_iris else "Closed"

            # ── 컬러 마스크 오버레이 (선택) ──
            overlay = frame.copy()
            for mask, cls in zip(masks, classes):
                col = COLORS[cls % len(COLORS)]
                mask = cv2.resize(mask.astype(np.uint8), (w, h),
                                  interpolation=cv2.INTER_NEAREST)
                for c in range(3):
                    overlay[:,:,c] = np.where(mask,
                        0.4*col[c] + 0.6*overlay[:,:,c], overlay[:,:,c])
            frame = overlay

        # 상태 바 추가
        frame = put_status_bar(frame, left_eye_state, right_eye_state)

        cv2.imshow("Eye State (4cls)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# 서버에서 되는지 확인 안해봄. 로컬에서만 돌려봄.
# PYTORCH_ENABLE_MPS_FALLBACK=1 python real_time.py
# mac M1/M2 GPU 사용시 PYTORCH_ENABLE_MPS_FALLBACK=1 해야 돌아감