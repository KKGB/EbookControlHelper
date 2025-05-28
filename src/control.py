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
    from AppKit import NSWorkspace
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


def focus_app_by_name(app_name):
    apps = NSWorkspace.sharedWorkspace().runningApplications()
    for app in apps:
        if app.localizedName().lower() == app_name.lower():
            app.activateWithOptions_(1 << 1)
            break

def _send_key(key: str, repeat: int = 1):
    for _ in range(repeat):
        if IS_WIN:
            keyboard.send(key)
        elif IS_MAC:
            if "+" in key:
                keys = key.split("+")
                for k in keys:
                    pyautogui.keyDown(k)
                for k in reversed(keys):
                    pyautogui.keyUp(k)
            else:
                pyautogui.press(key)

def control_pdf(gaze_state: int, process_name: str):
    global _last_command, pdf_mode

    if IS_WIN and process_name.lower() != "msedge.exe":
        if pdf_mode == "fit_page":
            return None, "PAGE MODE"
        else:
            return None, "SCROLL MODE"
    if IS_MAC and process_name.lower() != "microsoft edge":
        if pdf_mode == "fit_page":
            return None, "PAGE MODE"
        else:
            return None, "SCROLL MODE"
    if gaze_state == _last_command:
        if pdf_mode == "fit_page":
            return None, "PAGE MODE"
        else:
            return None, "SCROLL MODE"

    _last_command = gaze_state
    action_list = _gaze_actions.get(gaze_state)

    if gaze_state == 0 or gaze_state == 1 or gaze_state == 3 or gaze_state == 4:
        if gaze_state == 3:
            pdf_mode = "fit_page" if pdf_mode == "fit_width" else "fit_width"
        if gaze_state == 0 or gaze_state == 1 or gaze_state == 4:
            mode_idx = 0 if pdf_mode == "fit_page" else 1
            key_cmd, repeat = action_list[mode_idx]
            _send_key(key_cmd, repeat)
    elif gaze_state == 2:
        focus_app_by_name(process_name)
    
    if gaze_state == 0:
        if pdf_mode == "fit_page":
            return "PAGE DOWN", "PAGE MODE"
        else:
            return "SCROLL DOWN", "SCROLL MODE"
    elif gaze_state == 1:
        if pdf_mode == "fit_page":
            return "PAGE UP", "PAGE MODE"
        else:
            return "SCROLL UP", "SCROLL MODE"
    elif gaze_state == 2:
        if pdf_mode == "fit_page":
            return "CENTER", "PAGE MODE"
        else:
            return "CENTER", "SCROLL MODE"
    elif gaze_state == 3:
        if pdf_mode == "fit_page":
            return "PAGE MODE", "PAGE MODE"
        else:
            return "SCROLL MODE", "SCROLL MODE"
    elif gaze_state == 4:
        if pdf_mode == "fit_page":
            return "", "PAGE MODE"
        else:
            return "", "SCROLL MODE"


def control_youtube(gaze_state: int, process_name: str):
    pass

def control_web(gaze_state: int, process_name: str):
    pass

def control_ppt(gaze_state: int, process_name: str):
    pass
