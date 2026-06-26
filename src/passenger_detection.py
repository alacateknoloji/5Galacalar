# -*- coding: utf-8 -*-
"""
Passenger / seat occupancy module.

This version assumes the model is a person detector. It first finds person
objects in the frame, then uses ROI regions to classify which seat area they
occupy. If no person is detected inside a given ROI, that seat returns empty.
"""
import os

try:
    from src import utils
except ModuleNotFoundError:
    import utils

# >>> CONFIG: passenger weight file inside the models dir <<<
WEIGHT_FILE = "passenger_detection.pt"

VALID_SEATS = {"arka_koltuk_1", "arka_koltuk_2", "on_koltuk"}

# Person detector may output different labels; we only care about person-like detections.
PERSON_LABELS = {"person", "insan", "kişi", "human"}

CONF_THRESHOLD = 0.30


def _roi_for_seat(frame, seat_name):
    """Return a rectangular ROI for the requested seat area."""
    if frame is None:
        return None
    h, w = frame.shape[:2]

    if seat_name == "on_koltuk":
        x1, y1 = int(w * 0.10), int(h * 0.25)
        x2, y2 = int(w * 0.45), int(h * 0.85)
    elif seat_name in {"arka_koltuk_1", "arka_koltuk_2"}:
        x1, y1 = int(w * 0.50), int(h * 0.20)
        x2, y2 = int(w * 0.95), int(h * 0.85)
    else:
        return None

    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def _person_in_roi(frame, roi):
    """Check whether a person-like object exists inside the given ROI."""
    if frame is None or roi is None:
        return False
    x1, y1, x2, y2 = roi
    if x2 <= x1 or y2 <= y1:
        return False
    roi_frame = frame[y1:y2, x1:x2]
    if roi_frame.size == 0:
        return False
    return True


def load_model(models_dir):
    return utils.load_yolo(os.path.join(models_dir, WEIGHT_FILE))


def detect(model, frame, device):
    """Return seat occupancy detections based on person detections inside ROIs."""
    if model is None or frame is None:
        return []
    try:
        result = model(frame, conf=CONF_THRESHOLD, device=device, verbose=False)[0]
        dets = utils.detections_from_result(result, conf_threshold=CONF_THRESHOLD)
        out = []

        for seat_name in ["on_koltuk", "arka_koltuk_1", "arka_koltuk_2"]:
            roi = _roi_for_seat(frame, seat_name)
            if not roi:
                continue

            occupied = False
            for d in dets:
                raw = utils.to_ascii(d["label"])
                if raw in PERSON_LABELS:
                    x1, y1, x2, y2 = d["bbox"]
                    if x2 >= roi[0] and x1 <= roi[2] and y2 >= roi[1] and y1 <= roi[3]:
                        occupied = True
                        break

            if occupied:
                out.append({"label": seat_name, "conf": 0.85})

        return out
    except Exception:
        return []
