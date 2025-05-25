import sys
import cv2
import torch
import numpy as np
import platform
from collections import deque, Counter
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QDesktopWidget, QGraphicsOpacityEffect
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation
from PyQt5.QtGui import QFont
from ultralytics import YOLO

# ────────────────────────────── 시선 추적 스레드 ──────────────────────────────
class EyeTrackerThread(QThread):
    gaze_updated = pyqtSignal(str)

    def __init__(self, model_path="models/best.pt", cam_id=0):
        super().__init__()
        self.model = YOLO(model_path)
        self.model.fuse()
        self.cap = cv2.VideoCapture(cam_id)
        self.running = True

        if platform.system() == "Darwin":
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.required_frames = 30  # 예: 3초 @ 10fps
        self.min_agreement = int(self.required_frames * 0.9)
        self.direction_buffer = deque(maxlen=self.required_frames)
        self.gaze_directions = {0: "Right", 1: "Left", 2: "Center"}
        self.confirmed_gaze = None

    def get_center(self, mask):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        return int(xs.mean()), int(ys.mean())

    def run(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            res = self.model(frame, imgsz=640, conf=0.25, iou=0.3, device=self.device, verbose=False)[0]
            iris_mask = lid_mask = None

            if res.masks is not None:
                masks = (res.masks.data > 0.5).cpu().numpy()
                classes = res.boxes.cls.int().cpu().tolist()

                for mask, cls in zip(masks, classes):
                    if cls == 1: iris_mask = mask  # right_iris
                    if cls == 3: lid_mask  = mask  # right_eyelid

                if iris_mask is not None and lid_mask is not None:
                    iris_c = self.get_center(iris_mask)
                    lid_c  = self.get_center(lid_mask)
                    if iris_c and lid_c:
                        dx = iris_c[0] - lid_c[0]
                        if dx > 5:
                            current_gaze = 0
                        elif dx < -5:
                            current_gaze = 1
                        else:
                            current_gaze = 2
                        self.direction_buffer.append(current_gaze)

                        if len(self.direction_buffer) == self.required_frames:
                            counts = Counter(self.direction_buffer)
                            most_common, count = counts.most_common(1)[0]
                            if count >= self.min_agreement and most_common != self.confirmed_gaze:
                                self.confirmed_gaze = most_common
                                self.gaze_updated.emit(self.gaze_directions[most_common])
                            self.direction_buffer.clear()

        self.cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

# ────────────────────────────── 오버레이 창 ──────────────────────────────
class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        # 전체 창 크기 (화면 전체로 설정)
        screen_rect = QDesktopWidget().availableGeometry()
        self.setGeometry(screen_rect)

        # 라벨 설정
        self.label = QLabel("Gaze: Detecting...", self)
        self.label.setFont(QFont("Arial", 36, QFont.Bold))
        self.label.setStyleSheet("color: yellow; background-color: transparent;")

        # 투명도 효과
        self.opacity_effect = QGraphicsOpacityEffect(self.label)
        self.label.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(0.0)

        self.label.adjustSize()
        self.label.hide()

        # 애니메이션
        self.fade_anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_anim.setDuration(1000)  # 1초 동안 fade-out

        # 사라진 뒤 자동 숨김
        self.fade_anim.finished.connect(self.label.hide)

        self.show()

    def update_gaze(self, gaze_text):
        # 텍스트 변경 및 위치 조정
        self.label.setText(f"Gaze: {gaze_text}")
        self.label.adjustSize()
        screen_rect = QDesktopWidget().availableGeometry()
        screen_width = screen_rect.width()
        screen_height = screen_rect.height()
        label_width = self.label.width()
        label_height = self.label.height()

        if "Left" in gaze_text:
            x = int(screen_width * 0.1)
        elif "Right" in gaze_text:
            x = int(screen_width * 0.9 - label_width)
        else:
            x = (screen_width - label_width) // 2
        y = (screen_height - label_height) // 2
        self.label.move(x, y)

        # 즉시 표시 및 불투명하게
        self.fade_anim.stop()
        self.opacity_effect.setOpacity(1.0)
        self.label.show()

        # 1초 후 fade-out 시작
        QTimer.singleShot(1000, self.start_fade_out)

    def start_fade_out(self):
        self.fade_anim.setStartValue(1.0)
        self.fade_anim.setEndValue(0.0)
        self.fade_anim.start()

# ────────────────────────────── 메인 실행 ──────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)

    overlay = OverlayWindow()
    tracker = EyeTrackerThread()
    tracker.gaze_updated.connect(overlay.update_gaze)

    tracker.start()
    exit_code = app.exec_()
    tracker.stop()
    sys.exit(exit_code)
