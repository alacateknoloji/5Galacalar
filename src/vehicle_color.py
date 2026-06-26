# -*- coding: utf-8 -*-
"""
Vehicle colour module.

Classifies the body colour of the main vehicle from its cropped region.
Works with both a YOLO classification model (result.probs) and a YOLO
detection model (highest-confidence box) via utils.top_class_from_result.
"""

import os

try:
    from src import utils
except ModuleNotFoundError:
    import utils

# >>> CONFIG: colour weight file inside the models dir <<<
WEIGHT_FILE = "vehicle_color.pt"

VALID_COLORS = {"beyaz", "siyah", "gri", "kirmizi", "mavi",
                "sari", "yesil", "turuncu", "kahverengi"}

# >>> CONFIG: map YOUR colour model's class names -> competition colours <<<
LABEL_MAP = {
    "beyaz": "beyaz", "siyah": "siyah", "gri": "gri", "kirmizi": "kirmizi",
    "mavi": "mavi", "sari": "sari", "yesil": "yesil", "turuncu": "turuncu",
    "kahverengi": "kahverengi",
    # english fallbacks:
    # "white": "beyaz", "black": "siyah", "gray": "gri", "grey": "gri",
    # "red": "kirmizi", "blue": "mavi", "yellow": "sari", "green": "yesil",
    # "orange": "turuncu", "brown": "kahverengi",
}

CONF_THRESHOLD = 0.20


def load_model(models_dir):
    return utils.load_yolo(os.path.join(models_dir, WEIGHT_FILE))


def classify(model, crop, device):
    """
    Classify the colour of a cropped vehicle image.
    Returns (color_or_None, confidence). Safe default (None, 0.0) on failure.
    """
    if model is None or crop is None or crop.size == 0:
        return None, 0.0
    try:
        result = model(crop, device=device, verbose=False)[0]
        raw, conf = utils.top_class_from_result(result)
        if raw is None or conf < CONF_THRESHOLD:
            return None, 0.0
        mapped = LABEL_MAP.get(utils.to_ascii(raw), LABEL_MAP.get(raw, utils.to_ascii(raw)))
        return (mapped, conf) if mapped in VALID_COLORS else (None, 0.0)
    except Exception:
        return None, 0.0
