REGISTRY = {}

from .control import control_pdf
from .control import control_youtube
from .control import control_web
from .control import control_ppt

REGISTRY['pdf'] = control_pdf
REGISTRY['youtube'] = control_youtube
REGISTRY['web'] = control_web
REGISTRY['ppt'] = control_ppt