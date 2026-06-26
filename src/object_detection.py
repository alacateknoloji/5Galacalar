#   -*- coding: utf-8 -*-
"""
Object detection module.

Detects in-cabin / on-road target objects. Only two output labels are valid:
'teknocan' and 'bilgisayar'. Category downstream is always "nesneler".
"""

import os
try:
    from src import utils
except ModuleNotFoundError:
    import utils

# >>> CONFIG: object weight file inside the models dir <<<
WEIGHT_FILE = "object_detection.pt"

VALID_OBJECTS = {"teknocan", "bilgisayar", "insan"}

# >>> CONFIG: map YOUR model's class names -> competition object labels <<<
LABEL_MAP = {
    "teknocan": "teknocan", "bilgisayar": "bilgisayar", "insan": "insan",
    # english fallbacks:
    # "laptop": "bilgisayar", "computer": "bilgisayar", "can": "teknocan",
}

CONF_THRESHOLD = 0.30


def _get_detection_frame(frame, bbox):
    """Return the frame crop corresponding to the vehicle bbox when provided."""
    if frame is None:
        return None
    if bbox is None:
        return frame
    return utils.crop_bbox(frame, bbox, pad=0.0)


def load_model(models_dir):
    return utils.load_yolo(os.path.join(models_dir, WEIGHT_FILE))


def detect(model, frame, device, bbox=None):
    """Return [{"label": <valid object>, "conf": float}, ...] (empty if disabled)."""
    if model is None or frame is None:
        return []
    try:
        roi_frame = _get_detection_frame(frame, bbox)
        if roi_frame is None or roi_frame.size == 0:
            return []
        result = model(roi_frame, conf=CONF_THRESHOLD, device=device, verbose=False)[0]
        dets = utils.detections_from_result(result, conf_threshold=CONF_THRESHOLD)
        out = []
        for d in dets:
            raw = utils.to_ascii(d["label"])
            mapped = LABEL_MAP.get(raw, raw)
            if mapped in VALID_OBJECTS:
                out.append({"label": mapped, "conf": d["confidence"]})
        return out
    except Exception:
        return []
