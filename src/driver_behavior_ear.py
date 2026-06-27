# -*- coding: utf-8 -*-
"""
EAR-based driver drowsiness / yawning detection.

Model: driver_behavior_EAR.pt
Classes: closed_eye, closed_mouth, face, open_eye, open_mouth

Detection logic:
  - esneme    <- open_mouth detected within driver face ROI
  - uykululuk <- closed_eye detected within driver face ROI

ROI is computed relative to the car bounding box (upper driver-side area).
When a face detection is present it is used as an anchor so that only features
inside that face are counted, reducing false positives from background.
"""

import os

try:
    from src import utils
except ModuleNotFoundError:
    import utils

WEIGHT_FILE = "driver_behavior_EAR.pt"

CONF_THRESHOLD = 0.30

# Face ROI as fractions of car_bbox: right (driver) side, upper portion
_FACE_ROI_FRACS = (0.50, 0.05, 0.98, 0.52)


def _get_face_roi(frame, car_bbox=None):
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if car_bbox is not None:
        cx1, cy1, cx2, cy2 = car_bbox
        cw, ch = cx2 - cx1, cy2 - cy1
    else:
        cx1, cy1, cw, ch = 0, 0, w, h

    fx1, fy1, fx2, fy2 = _FACE_ROI_FRACS
    x1 = max(0, int(cx1 + cw * fx1))
    y1 = max(0, int(cy1 + ch * fy1))
    x2 = min(w, int(cx1 + cw * fx2))
    y2 = min(h, int(cy1 + ch * fy2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _bbox_contains(outer, inner, overlap=0.5):
    """True if inner bbox overlaps outer by at least `overlap` fraction of inner's area."""
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    inter_x = max(0.0, min(ox2, ix2) - max(ox1, ix1))
    inter_y = max(0.0, min(oy2, iy2) - max(oy1, iy1))
    inter_area = inter_x * inter_y
    inner_area = max(1.0, (ix2 - ix1) * (iy2 - iy1))
    return (inter_area / inner_area) >= overlap


def _analyze(dets):
    """
    Interpret facial feature detections.
    If face(s) are detected, restrict feature analysis to within each face bbox.
    Returns list of {"label": str, "conf": float}.
    """
    faces = [d for d in dets if d["label"] == "face"]
    features = [d for d in dets if d["label"] != "face"]

    if faces:
        # Use the highest-confidence face as anchor
        anchor = max(faces, key=lambda d: d["confidence"])
        features = [d for d in features if _bbox_contains(anchor["bbox"], d["bbox"])]

    out = []

    closed_eyes = [d for d in features if d["label"] == "closed_eye"]
    open_mouths = [d for d in features if d["label"] == "open_mouth"]

    if closed_eyes:
        conf = max(d["confidence"] for d in closed_eyes)
        out.append({"label": "uykululuk", "conf": conf})

    if open_mouths:
        conf = max(d["confidence"] for d in open_mouths)
        out.append({"label": "esneme", "conf": conf})

    return out


def load_model(models_dir):
    return utils.load_yolo(os.path.join(models_dir, WEIGHT_FILE))


def detect(model, frame, device, car_bbox=None):
    """
    Run EAR-based analysis on the driver face ROI.
    Returns [{"label": "esneme"|"uykululuk", "conf": float}, ...].
    Returns [] when model is unavailable or no relevant features are found.
    """
    if model is None or frame is None:
        return []
    try:
        roi = _get_face_roi(frame, car_bbox)
        if roi is None:
            return []
        rx1, ry1, rx2, ry2 = roi
        roi_frame = frame[ry1:ry2, rx1:rx2]
        if roi_frame is None or roi_frame.size == 0:
            return []

        result = model(roi_frame, conf=CONF_THRESHOLD, device=device, verbose=False)[0]
        dets = utils.detections_from_result(result, conf_threshold=CONF_THRESHOLD)
        return _analyze(dets)
    except Exception:
        return []
