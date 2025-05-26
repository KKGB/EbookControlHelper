import keyboard
import yaml
import os
from utils.path import resource_path

def _load_gaze_actions(config_path="keymap/config.yaml"):
    config_path = resource_path(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Gaze config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

_gaze_actions = _load_gaze_actions()
_last_command = None

def control_pdf(gaze_state: int, process_name: str):
    global _last_command

    if process_name.lower() != "msedge.exe":
        return

    action = _gaze_actions.get(gaze_state)

    if action is None:
        _last_command = None
        return

    if _last_command == gaze_state:
        return

    _last_command = gaze_state

    key, repeat = action
    for _ in range(repeat):
        keyboard.send(key)

def control_youtube(gaze_state: int, process_name: str):
    pass

def control_web(gaze_state: int, process_name: str):
    pass

def control_ppt(gaze_state: int, process_name: str):
    pass