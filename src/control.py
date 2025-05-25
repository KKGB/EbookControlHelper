import keyboard

# 내부 상태 저장용
_last_command = None

def control_ps(gaze_state: int, process_name: str):
    global _last_command

    if process_name.lower() != "msedge.exe":
        return  # 허용된 앱이 아니면 무시

    gaze_actions = {
        0: ("pagedown", 1),    # Right
        1: ("pageup", 1),      # Left
        4: ("down", 5),        # Right_Close
        3: ("up", 5),          # Left_Close
        5: ("ctrl+\\", 1),     # Close
        2: None                # Center → 초기화
    }

    action = gaze_actions.get(gaze_state)

    if action is None:
        _last_command = None
        return

    if _last_command == gaze_state:
        return  # 중복 방지

    _last_command = gaze_state

    key, repeat = action
    for _ in range(repeat):
        keyboard.send(key)
