import sys
import cv2
import torch
import numpy as np
import platform
import yaml
from collections import deque, Counter
import psutil

from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QDesktopWidget, QGraphicsOpacityEffect
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation
from PyQt5.QtGui import QFont
from ultralytics import YOLO

from src import REGISTRY
from utils.path import resource_path

IS_MAC = platform.system() == "Darwin"
IS_WIN = platform.system() == "Windows"

if IS_WIN:
    import win32gui
    import win32process
elif IS_MAC:
    from AppKit import NSWorkspace


def get_foreground_process_name():
    if IS_WIN:
        try:
            hwnd = win32gui.GetForegroundWindow()
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            return psutil.Process(pid).name()
        except Exception:
            return "N/A"
    elif IS_MAC:
        try:
            active_app = NSWorkspace.sharedWorkspace().frontmostApplication()
            return active_app.localizedName()
        except Exception:
            return "N/A"
    else:
        return "N/A"


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

        if IS_MAC:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.required_frames = 30
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
    
    def get_closed(self, iris_c, iris_mask):
        ys, _ = np.nonzero(iris_mask)
        if len(ys) == 0:
            return False
        
        ys = set(ys)
        if len(ys) < 3 and iris_c in ys:
            return True
        return False

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

                left_iris_mask = right_iris_mask = left_lid_mask = right_lid_mask = None
                has_left_iris = has_right_iris = False
                has_left_lid = has_right_lid = False

                for mask, cls in zip(masks, classes):
                    if cls == 0:
                        has_left_iris = True
                        left_iris_mask = mask
                    elif cls == 1:
                        has_right_iris = True
                        right_iris_mask = mask
                    elif cls == 2:
                        has_left_lid = True
                        left_lid_mask = mask
                    elif cls == 3:
                        has_right_lid = True
                        right_lid_mask = mask

                left_closed = has_right_lid and has_right_iris and not has_left_lid and not has_left_iris
                right_closed = has_left_lid and has_left_iris and not has_right_lid and not has_right_iris

                if left_closed:
                    current_gaze = 3
                elif right_closed:
                    current_gaze = 4
                elif (left_iris_mask is not None and left_lid_mask is not None) or (right_iris_mask is not None and right_lid_mask is not None):
                    if left_iris_mask is not None and left_lid_mask is not None:
                        iris_mask = left_iris_mask
                        lid_mask = left_lid_mask
                    elif right_iris_mask is not None and right_lid_mask is not None:
                        iris_mask = right_iris_mask
                        lid_mask = right_lid_mask
                    iris_c = self.get_center(iris_mask)
                    lid_c = self.get_center(lid_mask)
                    both_closed = self.get_closed(iris_c, iris_mask)
                    if both_closed:
                        current_gaze = 5
                        break
                    if iris_c and lid_c:
                        dx = iris_c[0] - lid_c[0]
                        if dx > 5:
                            current_gaze = 0
                        elif dx < -5:
                            current_gaze = 1
                        else:
                            current_gaze = 2
                else:
                    continue

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


class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()
        if IS_MAC:
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        screen = QApplication.primaryScreen()
        screen_rect = screen.geometry()
        self.setGeometry(screen_rect)

        self.label = QLabel("Gaze: Detecting...", self)
        self.label.setFont(QFont("Arial", 36, QFont.Bold))
        self.label.setStyleSheet("color: yellow; background-color: transparent;")

        self.opacity_effect = QGraphicsOpacityEffect(self.label)
        self.label.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(0.0)

        self.label.adjustSize()
        self.label.hide()

        self.fade_anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_anim.setDuration(1000)
        self.fade_anim.finished.connect(self.label.hide)

        self.show()

        self.border = QWidget(self)
        self.border.setGeometry(self.rect())
        self.border.setStyleSheet("""
            background-color: transparent;
            border: 5px solid limegreen;
        """)
        self.border.show()

        self.current_process_name = "N/A"
        self.proc_label = QLabel("Process: N/A", self)
        self.proc_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.proc_label.setStyleSheet("color: lightgreen; background-color: transparent;")
        self.proc_label.adjustSize()

        screen_width = screen_rect.width()
        proc_label_y = 40 if IS_MAC else 20
        self.proc_label.move(screen_width - self.proc_label.width() - 20, proc_label_y)
        self.proc_label.show()

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

        if gaze_text == "Left":
            x = int(screen_width * 0.1)
        elif gaze_text == "Right":
            x = int(screen_width * 0.9 - label_width)
        else:
            x = (screen_width - label_width) // 2

        y = (screen_height - label_height) // 2
        self.label.move(x, y)

        self.fade_anim.stop()
        self.opacity_effect.setOpacity(1.0)
        self.label.show()
        QTimer.singleShot(1000, self.start_fade_out)

    def update_process_name(self):
        self.current_process_name = get_foreground_process_name()
        self.proc_label.setText(f"Process: {self.current_process_name}")
        self.proc_label.adjustSize()

        screen = QApplication.primaryScreen()
        screen_width = screen.geometry().width()
        proc_label_y = 40 if IS_MAC else 20
        self.proc_label.move(screen_width - self.proc_label.width() - 20, proc_label_y)

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
        raise ValueError(f"❌ REGISTRY가 '{process_name}'에 해당하는 컨트롤러가 등록되어 있지 않습니다.\n"
                         f"가능한 키: {list(REGISTRY.keys())}")

    overlay = OverlayWindow()
    tracker = EyeTrackerThread(overlay=overlay, process_name=process_name)
    tracker.gaze_updated.connect(overlay.update_gaze)

    tracker.start()
    exit_code = app.exec_()
    tracker.stop()
    sys.exit(exit_code)
