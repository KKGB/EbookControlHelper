import sys
import os

def resource_path(relative_path):
    """PyInstaller에서 실행할 때 올바른 리소스 경로 반환"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)  # PyInstaller 빌드된 경우
    return os.path.join(os.path.abspath("."), relative_path)  # 개발 중 실행하는 경우