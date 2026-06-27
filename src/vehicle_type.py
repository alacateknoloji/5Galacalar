# -*- coding: utf-8 -*-
"""
Vehicle type module.

Owns the vehicle-detection model. Its raw detections are reused by the slalom
module (which needs every vehicle box, frame by frame), while classify_main()
picks the single main vehicle and maps its class to the competition whitelist.
"""

import os

try:
    from src import utils
except ModuleNotFoundError:
    import utils

# >>> CONFIG: vehicle-detection weight file inside the models dir <<<
# Change this filename if your weight is named differently.
WEIGHT_FILE = "vehicle_type.pt"

# Valid output types (do NOT add others; the grader rejects them).
VALID_TYPES = {"sedan", "suv", "hatchback", "pickup", "minibus", "panelvan", "kamyon"}

# >>> CONFIG: map YOUR model's class names -> competition types <<<
# Your detector (Araba_modeli) also has classes like 'ambulans'/'itfaiye' that
# are NOT valid output types. Map every model class here. Anything not mapped to
# a VALID_TYPE is still tracked for slalom but is ignored for the 'tip' field.
LABEL_MAP = {
    "sedan": "sedan",
    "suv": "suv",
    "hatchback": "hatchback",
    "pickup": "pickup",
    "minibus": "minibus",
    "panelvan": "panelvan",
    "kamyon": "kamyon",
    # tracked for slalom bbox but not a valid output type
    "ambulans": None,
    "itfaiye": None,
}

CONF_THRESHOLD = 0.25


def load_model(models_dir):
    return utils.load_yolo(os.path.join(models_dir, WEIGHT_FILE))


def detect(model, frame, device):
    """
    Run vehicle detection on a frame. Returns raw boxes for tracking:
        [{"label": <model class>, "bbox": [x1,y1,x2,y2], "confidence": float}, ...]
    Returns [] if the model is unavailable or inference fails.
    """
    if model is None:
        return []
    try:
        result = model(frame, conf=CONF_THRESHOLD, device=device, verbose=False)[0]
        return utils.detections_from_result(result, conf_threshold=CONF_THRESHOLD)
    except Exception:
        return []


def classify_main(detections):
    """
    Choose the main vehicle (largest box, tie-break by confidence) and map its
    class to a valid output type.
    Returns (tip_or_None, confidence, main_bbox_or_None).
    The bbox is returned even when the type is not mappable, so colour/plate
    modules can still use the crop.
    """
    if not detections:
        return None, 0.0, None

    main = max(detections, key=lambda d: (utils.bbox_area(d["bbox"]), d["confidence"]))
    raw = main["label"]
    mapped = LABEL_MAP.get(raw, raw)
    tip = mapped if mapped in VALID_TYPES else None
    return tip, main["confidence"], main["bbox"]
