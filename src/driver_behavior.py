# -*- coding: utf-8 -*-
"""
Driver behaviour module.

STATUS: the driver_behavior weight is NOT ready yet. Until the .pt file is
placed in the models dir, load_model() returns None and detect() returns [] (no
driver-action detections). The moment you drop a working driver_behavior.pt in,
this module activates automatically with no other code change.

This is honest degradation, not environment manipulation: the module simply
produces nothing while its model is absent.
"""

import os

try:
    from src import utils
except ModuleNotFoundError:
    import utils


PERSON_LABELS = {"person", "insan", "kisi", "human"}

# Driver seat occupies the right portion of the car bbox
# (right-hand side of the car interior image in Turkish vehicles)
_DRIVER_ROI_FRACS = (0.50, 0.15, 0.98, 0.92)


def _get_driver_roi(frame, car_bbox=None):
    """Return the driver seat ROI within the car bounding box."""
    if frame is None:
        return None
    h, w = frame.shape[:2]

    if car_bbox is not None:
        cx1, cy1, cx2, cy2 = car_bbox
        cw = cx2 - cx1
        ch = cy2 - cy1
    else:
        cx1, cy1, cw, ch = 0, 0, w, h

    fx1, fy1, fx2, fy2 = _DRIVER_ROI_FRACS
    x1 = max(0, int(cx1 + cw * fx1))
    y1 = max(0, int(cy1 + ch * fy1))
    x2 = min(w, int(cx1 + cw * fx2))
    y2 = min(h, int(cy1 + ch * fy2))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _bbox_center_x(bbox):
    x1, _, x2, _ = bbox
    return (x1 + x2) / 2.0


def _bbox_intersects_roi(bbox, roi):
    x1, y1, x2, y2 = bbox
    rx1, ry1, rx2, ry2 = roi
    return x2 >= rx1 and x1 <= rx2 and y2 >= ry1 and y1 <= ry2


def _select_driver_person(object_detections, roi=None):
    """Select the rightmost person detection, optionally restricted to a driver ROI."""
    if not object_detections:
        return None
    candidates = []
    for d in object_detections:
        raw = utils.to_ascii(d.get("label"))
        if raw in PERSON_LABELS and "bbox" in d:
            bbox = d["bbox"]
            if roi is None or _bbox_intersects_roi(bbox, roi):
                candidates.append(d)
    if not candidates:
        return None
    return max(candidates, key=lambda d: _bbox_center_x(d["bbox"]))

# >>> CONFIG: driver-behaviour weight file inside the models dir <<<
WEIGHT_FILE = "driver_behavior.pt"

# Valid driver-action labels (kategori is always "sofor_eylemi" downstream).
# NOTE: "slalom" is emitted by the slalom module from vehicle motion, not here.
VALID_ACTIONS = {
    "bir_sey_icme", "kemer_takili", "mesajlasma",
    "sigara_icme", "telefonla_konusma",
}

LABEL_MAP = {
    "bir_sey_icme":       "bir_sey_icme",
    "kemer_takili":       "kemer_takili",
    "mesajlasma":         "mesajlasma",
    "sigara_icme":        "sigara_icme",
    "telefonla_konusma":  "telefonla_konusma",
}

CONF_THRESHOLD = 0.30


def load_model(models_dir):
    return utils.load_yolo(os.path.join(models_dir, WEIGHT_FILE))


def detect(model, frame, device, object_detections=None, car_bbox=None):
    """
    Detect driver actions on a frame.
    ROI is computed relative to car_bbox when provided.
    If object_detections are available, the rightmost person within the driver
    ROI is selected and cropped; otherwise the ROI crop is used directly.
    """
    if model is None or frame is None:
        return []
    try:
        roi = _get_driver_roi(frame, car_bbox)
        driver_crop = None

        if object_detections is not None:
            selected = _select_driver_person(object_detections, roi=roi)
            if selected is not None:
                driver_crop = utils.crop_bbox(frame, selected["bbox"], pad=0.10)

        if driver_crop is None:
            if roi is None:
                return []
            x1, y1, x2, y2 = roi
            driver_crop = frame[y1:y2, x1:x2]

        if driver_crop is None or driver_crop.size == 0:
            return []

        result = model(driver_crop, conf=CONF_THRESHOLD, device=device, verbose=False)[0]
        dets = utils.detections_from_result(result, conf_threshold=CONF_THRESHOLD)
        out = []
        for d in dets:
            raw = utils.to_ascii(d["label"])
            mapped = LABEL_MAP.get(raw, raw)
            if mapped in VALID_ACTIONS:
                out.append({"label": mapped, "conf": d["confidence"]})
        return out
    except Exception:
        return []
