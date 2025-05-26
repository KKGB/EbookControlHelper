import sys
import cv2
import torch
import numpy as np
import platform
import yaml
from collections import deque, Counter
import psutil
import win32gui
import win32process
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QDesktopWidget, QGraphicsOpacityEffect
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation
from PyQt5.QtGui import QFont
from ultralytics import YOLO

from src import REGISTRY
from utils.path import resource_path


# ────────────────────────────── 시선 추적 스레드 ──────────────────────────────
class EyeTrackerThread(QThread):
    gaze_updated = pyqtSignal(str)

    def __init__(self, model_path="models/best.pt", cam_id=0, overlay=None, process_name=None):
        super().__init__()
        model_path = resource_path(model_path)
        self.model = YOLO(model_path)
        self.model.fuse()
        self.cap = cv2.VideoCapture(cam_id)
        self.running = True
        self.process_name = process_name

        if platform.system() == "Darwin":
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.required_frames = 30  # 예: 3초 @ 10fps
        self.min_agreement = int(self.required_frames * 0.9)
        self.direction_buffer = deque(maxlen=self.required_frames)
        self.gaze_directions = {0: "Right", 1: "Left", 2: "Center", 3: "Left_Close", 4: "Right_Close", 5: "Close"}
        self.confirmed_gaze = None
        self.overlay = overlay

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

            res = self.model(frame, imgsz=640, conf=0.25, iou=0.3, device=self.device, verbose=False)[0]
            iris_mask = lid_mask = None

            if res.masks is not None:
                masks = (res.masks.data > 0.5).cpu().numpy()
                classes = res.boxes.cls.int().cpu().tolist()

                # 초기 상태
                iris_mask = lid_mask = None
                has_left_iris = has_right_iris = False
                has_left_lid = has_right_lid = False

                for mask, cls in zip(masks, classes):
                    if cls == 0:
                        has_left_iris = True
                    elif cls == 1:
                        has_right_iris = True
                        iris_mask = mask  # right_iris
                    elif cls == 2:
                        has_left_lid = True
                    elif cls == 3:
                        has_right_lid = True
                        lid_mask = mask   # right_eyelid                        

                # 감은 눈 상태 판단
                both_closed = not has_left_lid and not has_left_iris and not has_right_lid and not has_right_iris
                left_closed = has_right_lid and has_right_iris and not has_left_lid and not has_left_iris
                right_closed = has_left_lid and has_left_iris and not has_right_lid and not has_right_iris
                
                if both_closed:
                    current_gaze = 5  # Close
                elif left_closed:
                    current_gaze = 3  # Left_Close
                elif right_closed:
                    current_gaze = 4  # Right_Close
                elif iris_mask is not None and lid_mask is not None:
                    iris_c = self.get_center(iris_mask)
                    lid_c = self.get_center(lid_mask)
                    if iris_c and lid_c:
                        dx = iris_c[0] - lid_c[0]
                        if dx > 5:
                            current_gaze = 0  # Right
                        elif dx < -5:
                            current_gaze = 1  # Left
                        else:
                            current_gaze = 2  # Center
                else:
                    continue  # 감지 실패

                self.direction_buffer.append(current_gaze)

                if len(self.direction_buffer) == self.required_frames:
                    counts = Counter(self.direction_buffer)
                    most_common, count = counts.most_common(1)[0]
                    if count >= self.min_agreement and most_common != self.confirmed_gaze:
                        REGISTRY[self.process_name](most_common, self.overlay.current_process_name)
                        self.confirmed_gaze = most_common
                        self.gaze_updated.emit(self.gaze_directions[most_common])
                        print("👁 Gaze:", self.gaze_directions[most_common])
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
        
        # 초록색 실행 상태 테두리 표시용 위젯
        self.border = QWidget(self)
        self.border.setGeometry(self.rect())
        self.border.setStyleSheet("""
            background-color: transparent;
            border: 5px solid limegreen;
        """)
        self.border.show()
        
        # 오른쪽 상단 프로세스 표시 라벨
        self.current_process_name = "N/A"
        self.proc_label = QLabel("Process: N/A", self)
        self.proc_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.proc_label.setStyleSheet("color: lightgreen; background-color: transparent;")
        self.proc_label.adjustSize()

        # 위치: 오른쪽 상단 여백 포함
        screen_width = screen_rect.width()
        self.proc_label.move(screen_width - self.proc_label.width() - 20, 20)
        self.proc_label.show()

        # 타이머: 1초마다 현재 프로세스 확인
        self.proc_timer = QTimer(self)
        self.proc_timer.timeout.connect(self.update_process_name)
        self.proc_timer.start(1000)

    def update_gaze(self, gaze_text):
        self.label.setText(f"Gaze: {gaze_text}")
        self.label.adjustSize()

        screen_rect = QDesktopWidget().availableGeometry()
        screen_width = screen_rect.width()
        screen_height = screen_rect.height()
        label_width = self.label.width()
        label_height = self.label.height()

        # 위치 결정 로직 업데이트
        if gaze_text == "Left":
            x = int(screen_width * 0.1)
        elif gaze_text == "Right":
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
    
    def update_process_name(self):
        try:
            hwnd = win32gui.GetForegroundWindow()
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            process_name = psutil.Process(pid).name()
            self.current_process_name = process_name  # 여기 추가
            self.proc_label.setText(f"Process: {process_name}")
            self.proc_label.adjustSize()
            screen_rect = QDesktopWidget().availableGeometry()
            screen_width = screen_rect.width()
            self.proc_label.move(screen_width - self.proc_label.width() - 20, 20)
        except Exception:
            self.current_process_name = "N/A"
            self.proc_label.setText("Process: N/A")
    
    def start_fade_out(self):
        self.fade_anim.setStartValue(1.0)
        self.fade_anim.setEndValue(0.0)
        self.fade_anim.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    config_path = resource_path("keymap/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    process_name = config.get("control")

    if process_name not in REGISTRY:
        raise ValueError(f"❌ REGISTRY에 '{process_name}'에 해당하는 컨트롤러가 등록되어 있지 않습니다.\n"
                         f"가능한 키: {list(REGISTRY.keys())}")

    overlay = OverlayWindow()
    tracker = EyeTrackerThread(overlay=overlay, process_name=process_name)
    tracker.gaze_updated.connect(overlay.update_gaze)

    tracker.start()
    exit_code = app.exec_()
    tracker.stop()
    sys.exit(exit_code)
