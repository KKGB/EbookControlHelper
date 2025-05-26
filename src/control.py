import os
import platform
import yaml
from utils.path import resource_path

IS_MAC = platform.system() == "Darwin"
IS_WIN = platform.system() == "Windows"

if IS_WIN:
    import keyboard as win_keyboard
elif IS_MAC:
    import pyautogui


def _load_gaze_actions(config_path="keymap/config.yaml"):
    config_path = resource_path(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Gaze config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

_gaze_actions = _load_gaze_actions()
_last_command = None

def send_key(key: str, repeat: int):
    for _ in range(repeat):
        if IS_WIN:
            win_keyboard.send(key)
        elif IS_MAC:
            if "+" in key:
                keys = key.split("+")
                pyautogui.hotkey(*keys)
            else:
                pyautogui.press(key)

def control_pdf(gaze_state: int, process_name: str):
    global _last_command

    target_process = process_name.lower()
    if (IS_WIN and target_process != "msedge.exe") or (IS_MAC and target_process != "microsoft edge"):
        return

    action = _gaze_actions.get(gaze_state)
    if action is None:
        _last_command = None
        return

    if _last_command == gaze_state:
        return

    _last_command = gaze_state
    key, repeat = action
    send_key(key, repeat)

def control_youtube(gaze_state: int, process_name: str):
    pass

def control_web(gaze_state: int, process_name: str):
    pass

def control_ppt(gaze_state: int, process_name: str):
    pass