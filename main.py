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
from PyQt5.QtGui import QFont, QImage, QPixmap
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

COLORS = [
    (255,  64,  64),  # right_iris
    ( 64,  64, 255),  # left_iris
    (255, 192,   0),  # right_eyelid
    (  0, 255, 128),  # left_eyelid
]

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
    preview_frame = pyqtSignal(np.ndarray)

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

        self.required_frames = 10
        self.min_agreement = int(self.required_frames * 0.8)
        self.direction_buffer = deque(maxlen=self.required_frames)
        self.gaze_directions = {0: "Right", 1: "Left", 2: "Center", 3: "Left_Close", 4: "Right_Close"}
        self.confirmed_gaze = None
        self.overlay = overlay

    def get_center(self, mask):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        return int(xs.mean()), int(ys.mean())

    def detect_gaze(self, masks, classes):
        mask_dict = {cls: mask for mask, cls in zip(masks, classes)}
        left_iris = mask_dict.get(0)
        right_iris = mask_dict.get(1)
        left_lid = mask_dict.get(2)
        right_lid = mask_dict.get(3)

        left_iris_c = right_iris_c = left_lid_c = right_lid_c = None
        left_iris_mask = right_iris_mask = left_lid_mask = right_lid_mask = None

        if left_iris is not None and left_lid is not None:
            left_iris_mask, left_lid_mask = left_iris, left_lid
            left_iris_c = self.get_center(left_iris_mask)
            left_lid_c = self.get_center(left_lid_mask)
        if right_iris is not None and right_lid is not None:
            right_iris_mask, right_lid_mask = right_iris, right_lid
            right_iris_c = self.get_center(right_iris_mask)
            right_lid_c = self.get_center(right_lid_mask)

        if right_lid is not None and right_iris is not None and (left_lid is None and left_iris is None):
            return 3  # Right_Close
        if left_lid is not None and left_iris is not None and (right_lid is None and right_iris is None):
            return 4  # Left_Close

        dx_values = []
        if left_iris_c is not None and left_lid_c is not None:
            dx_values.append(left_iris_c[0] - left_lid_c[0])
        if right_iris_c is not None and right_lid_c is not None:
            dx_values.append(right_iris_c[0] - right_lid_c[0])

        if not dx_values:
            return None

        dx_avg = np.mean(dx_values)
        if dx_avg > 4:
            return 0  # Right
        elif dx_avg < -4:
            return 1  # Left
        else:
            return 2  # Center

    def run(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            res = self.model(frame, imgsz=640, conf=0.25, iou=0.3, device=self.device, verbose=False)[0]

            if res.masks is not None:
                masks = (res.masks.data > 0.5).cpu().numpy()
                classes = res.boxes.cls.int().cpu().tolist()
                current_gaze = self.detect_gaze(masks, classes)

                if current_gaze is not None:
                    self.direction_buffer.append(current_gaze)
                    if len(self.direction_buffer) == self.required_frames:
                        counts = Counter(self.direction_buffer)
                        most_common, count = counts.most_common(1)[0]
                        if count >= self.min_agreement and most_common != self.confirmed_gaze:
                            title = REGISTRY[self.process_name](most_common, self.overlay.current_process_name)
                            self.confirmed_gaze = most_common
                            self.gaze_updated.emit(title)
                            print("👁 Gaze:", self.gaze_directions[most_common])
                        self.direction_buffer.clear()
                
                overlay = frame.copy()
                for mask, cls in zip(masks, classes):
                    col = COLORS[cls % len(COLORS)]
                    mask = cv2.resize(mask.astype(np.uint8), (w, h),
                                    interpolation=cv2.INTER_NEAREST)
                    for c in range(3):
                        overlay[:,:,c] = np.where(mask,
                            0.4*col[c] + 0.6*overlay[:,:,c], overlay[:,:,c])
                frame = overlay

            self.preview_frame.emit(frame)
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
        self.label.setStyleSheet("color: red; background-color: transparent;")

        self.opacity_effect = QGraphicsOpacityEffect(self.label)
        self.label.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(0.0)

        self.label.adjustSize()
        self.label.hide()

        self.fade_anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_anim.setDuration(1000)
        self.fade_anim.finished.connect(self.label.hide)

        self.preview_label = QLabel(self)
        self.preview_label.setFixedSize(320, 240)  # 2배로 확대
        self.preview_label.move(self.width() - 340, self.height() - 260)  # 위치도 조정
        self.preview_label.setStyleSheet("border: 2px solid white; background-color: black;")
        self.preview_label.hide()

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
        self.label.setText(gaze_text)
        self.label.adjustSize()

        screen_rect = QDesktopWidget().availableGeometry()
        screen_width = screen_rect.width()
        screen_height = screen_rect.height()
        label_width = self.label.width()
        label_height = self.label.height()

        if gaze_text == "SCROLL UP":
            x = int(screen_width * 0.1)
        elif gaze_text == "SCROLL DOWN":
            x = int(screen_width * 0.9 - label_width)
        else:
            x = (screen_width - label_width) // 2

        y = (screen_height - label_height) // 2
        self.label.move(x, y)

        self.fade_anim.stop()
        self.opacity_effect.setOpacity(1.0)
        self.label.show()
        QTimer.singleShot(1000, self.start_fade_out)

    def update_preview(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(320, 240, Qt.KeepAspectRatio)
        self.preview_label.setPixmap(scaled_pixmap)
        self.preview_label.show()

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
    tracker.preview_frame.connect(overlay.update_preview)

    tracker.start()
    exit_code = app.exec_()
    tracker.stop()
    sys.exit(exit_code)