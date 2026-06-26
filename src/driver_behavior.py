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


DEFAULT_ROI_RATIO = 0.35


def _get_driver_roi(frame):
    """Return a ROI on the right side of the front-seat area."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    roi_w = max(80, int(round(w * DEFAULT_ROI_RATIO)))
    roi_h = max(80, int(round(h * 0.65)))

    front_seat_x1, front_seat_y1 = int(w * 0.10), int(h * 0.25)
    front_seat_x2, front_seat_y2 = int(w * 0.45), int(h * 0.85)

    x1 = max(0, int(round(front_seat_x2 + (front_seat_x2 - front_seat_x1) * 0.15)))
    y1 = max(0, int(round(front_seat_y1 + (front_seat_y2 - front_seat_y1) * 0.15)))
    x2 = min(w, x1 + roi_w)
    y2 = min(h, y1 + roi_h)
    return x1, y1, x2, y2

# >>> CONFIG: driver-behaviour weight file inside the models dir <<<
WEIGHT_FILE = "driver_behavior.pt"

# Valid driver-action labels (kategori is always "sofor_eylemi" downstream).
# NOTE: "slalom" is emitted by the slalom module from vehicle motion, not here.
VALID_ACTIONS = {
    "arkaya_bakma", "esneme", "sigara_icme", "su_icme",
    "telefonla_konusma", "etrafa_bakinma", "bir_sey_icme", "kemer_takili",
    "mesajlasma",
}

# >>> CONFIG: map YOUR model's class names -> competition action labels <<<
LABEL_MAP = {
    "arkaya_bakma": "arkaya_bakma", "esneme": "esneme", "sigara_icme": "sigara_icme",
    "su_icme": "su_icme", "telefonla_konusma": "telefonla_konusma",
    "etrafa_bakinma": "etrafa_bakinma", "bir_sey_icme": "bir_sey_icme",
    "kemer_takili": "kemer_takili", "mesajlasma": "mesajlasma",
    # Seat-belt logic: if your model outputs a "belt on" class, map it to None
    # (no violation). Only emit emniyet_kemeri_ihlali when the belt is absent.
    # "no_seatbelt": "emniyet_kemeri_ihlali", "seatbelt": None,
    # "phone": "telefonla_konusma", "smoking": "sigara_icme", "yawning": "esneme",
}

CONF_THRESHOLD = 0.30


def load_model(models_dir):
    return utils.load_yolo(os.path.join(models_dir, WEIGHT_FILE))


def detect(model, frame, device):
    """
    Detect driver actions on a frame.
    Returns [{"label": <valid action>, "conf": float}, ...]. Empty while the
    model is unavailable.
    """
    if model is None or frame is None:
        return []
    try:
        roi = _get_driver_roi(frame)
        if roi is None:
            return []
        x1, y1, x2, y2 = roi
        roi_frame = frame[y1:y2, x1:x2]
        if roi_frame.size == 0:
            return []
        result = model(roi_frame, conf=CONF_THRESHOLD, device=device, verbose=False)[0]
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
