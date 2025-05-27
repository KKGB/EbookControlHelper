import os
import yaml
import platform
from utils.path import resource_path

IS_WIN = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"

if IS_WIN:
    import keyboard
elif IS_MAC:
    import pyautogui
pdf_mode = "fit_page"
_last_command = None


def _load_gaze_actions(config_path="keymap/config.yaml"):
    config_path = resource_path(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Gaze config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

_gaze_config = _load_gaze_actions()
_gaze_actions = {k: v for k, v in _gaze_config.items() if isinstance(k, int)}


def _send_key(key: str, repeat: int = 1):
    for _ in range(repeat):
        if IS_WIN:
            keyboard.send(key)
        elif IS_MAC:
            pyautogui.press(key)


def control_pdf(gaze_state: int, process_name: str):
    global _last_command, pdf_mode

    if IS_WIN and process_name.lower() != "msedge.exe":
        return
    if IS_MAC and process_name.lower() != "microsoft edge":
        return

    action_list = _gaze_actions.get(gaze_state)
    if gaze_state == _last_command:
        return

    _last_command = gaze_state

    mode_idx = 0 if pdf_mode == "fit_page" else 1
    if gaze_state == 5:
        pdf_mode = "fit_width"
    if action_list[mode_idx] is None:
        return

    key_cmd, repeat = action_list[mode_idx]
    _send_key(key_cmd, repeat)


def control_youtube(gaze_state: int, process_name: str):
    pass

def control_web(gaze_state: int, process_name: str):
    pass

def control_ppt(gaze_state: int, process_name: str):
    pass
