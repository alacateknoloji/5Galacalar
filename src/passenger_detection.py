# -*- coding: utf-8 -*-
"""
Passenger / seat occupancy module.

No dedicated passenger model is required. Seat occupancy is determined by
checking which ROI region (relative to the detected car bbox) each person
detection from object_detection falls into.
"""

try:
    from src import utils
except ModuleNotFoundError:
    import utils

VALID_SEATS = {"arka_koltuk_1", "arka_koltuk_2", "on_koltuk"}

PERSON_LABELS = {"person", "insan", "kisi", "human"}

# Seat ROI boundaries as fractions of the car bounding box
# on_koltuk (front passenger): left portion of car
# arka_koltuk_1 / arka_koltuk_2: right back half, split into left/right
_SEAT_ROIS = {
    "on_koltuk":     (0.05, 0.20, 0.45, 0.90),
    "arka_koltuk_1": (0.50, 0.15, 0.72, 0.90),
    "arka_koltuk_2": (0.72, 0.15, 0.98, 0.90),
}


def _roi_for_seat(frame, seat_name, car_bbox=None):
    """Return absolute pixel ROI for the requested seat, optionally within car_bbox."""
    if frame is None:
        return None
    h, w = frame.shape[:2]

    if car_bbox is not None:
        cx1, cy1, cx2, cy2 = car_bbox
        cw = cx2 - cx1
        ch = cy2 - cy1
    else:
        cx1, cy1, cw, ch = 0, 0, w, h

    fracs = _SEAT_ROIS.get(seat_name)
    if fracs is None:
        return None

    fx1, fy1, fx2, fy2 = fracs
    x1 = max(0, int(cx1 + cw * fx1))
    y1 = max(0, int(cy1 + ch * fy1))
    x2 = min(w, int(cx1 + cw * fx2))
    y2 = min(h, int(cy1 + ch * fy2))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def load_model(models_dir):
    """No passenger model needed; returns None unconditionally."""
    return None


def detect(model, frame, device, object_detections=None, car_bbox=None):
    """
    Determine seat occupancy from person detections within car ROI regions.

    Uses bboxes from object_detections (persons detected by the object detection
    model). car_bbox anchors the seat ROIs to the detected vehicle region.
    Each person is assigned to at most one seat (first match wins).
    """
    if frame is None or not object_detections:
        return []
    try:
        persons = [
            d for d in object_detections
            if utils.to_ascii(d.get("label")) in PERSON_LABELS and "bbox" in d
        ]
        if not persons:
            return []

        out = []
        assigned = set()
        for seat_name in ["on_koltuk", "arka_koltuk_1", "arka_koltuk_2"]:
            roi = _roi_for_seat(frame, seat_name, car_bbox)
            if roi is None:
                continue
            rx1, ry1, rx2, ry2 = roi
            for i, d in enumerate(persons):
                if i in assigned:
                    continue
                px1, py1, px2, py2 = d["bbox"]
                if px2 >= rx1 and px1 <= rx2 and py2 >= ry1 and py1 <= ry2:
                    out.append({"label": seat_name, "conf": d.get("conf", 0.85)})
                    assigned.add(i)
                    break

        return out
    except Exception:
        return []
