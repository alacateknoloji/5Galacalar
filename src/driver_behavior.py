# -*- coding: utf-8 -*-
"""
Driver behaviour detection (action + EAR merged).

Models:
  - driver_behavior.pt     : action detection (bir_sey_icme, kemer_takili,
                             mesajlasma, sigara_icme, telefonla_konusma,
                             arkaya_bakma, etrafa_bakinma)
  - driver_behavior_EAR.pt : EAR-based detection (open_mouth→esneme, closed_eye→uykululuk)

Competition label mapping:
  bir_sey_icme      → su_icme
  kemer_takili      → absence triggers emniyet_kemeri_ihlali (inversion)
  mesajlasma        → detected but filtered (no competition equivalent)
  uykululuk         → detected but filtered by formatter (not in competition schema)
  sigara_icme       → sigara_icme
  telefonla_konusma → telefonla_konusma
  arkaya_bakma      → arkaya_bakma
  etrafa_bakinma    → etrafa_bakinma
  esneme            → esneme

load_model() returns {"action": model, "ear": model}; either may be None if
the weight file is absent — the module degrades gracefully without it.
detect() combines results from both models.
"""

import os

try:
    from src import utils
except ModuleNotFoundError:
    import utils

# ── weight files ─────────────────────────────────────────────────────────────

ACTION_WEIGHT_FILE = "driver_behavior.pt"
EAR_WEIGHT_FILE    = "driver_behavior_EAR.pt"

CONF_THRESHOLD = 0.30

# ── action-detection config ───────────────────────────────────────────────────

PERSON_LABELS = {"person", "insan", "kisi", "human"}

# Driver seat: right side of the car bbox interior
_DRIVER_ROI_FRACS = (0.50, 0.15, 0.98, 0.92)

VALID_ACTIONS = {
    "su_icme", "sigara_icme", "telefonla_konusma",
    "arkaya_bakma", "etrafa_bakinma",
}

# Maps model class names → competition label names.
# kemer_takili is handled via inversion in detect(): presence → no violation,
# absence when driver is visible → emniyet_kemeri_ihlali added directly.
# bir_sey_icme (drinking anything) → su_icme (competition equivalent)
# mesajlasma has no competition equivalent; omitted so it gets filtered.
LABEL_MAP = {
    "bir_sey_icme":      "su_icme",
    "sigara_icme":       "sigara_icme",
    "telefonla_konusma": "telefonla_konusma",
}

# ── EAR config ────────────────────────────────────────────────────────────────

# Face ROI: right (driver) side, upper portion of car bbox
_FACE_ROI_FRACS = (0.50, 0.05, 0.98, 0.52)

# Gözler bu kadar saniye kesintisiz kapalı kalırsa → uykululuk
CLOSED_EYE_THRESHOLD = 2.0


class _DrowsinessTracker:
    """Kapalı göz süresini takip eder; eşik aşılınca uykululuk üretir."""

    def __init__(self, threshold=CLOSED_EYE_THRESHOLD):
        self.threshold = threshold
        self._closed_since = None  # ilk kapanma zamanı (saniye)

    def update(self, closed_detected, timestamp):
        """
        closed_detected: o karedeki EAR modelinden closed_eye tespiti var mı.
        timestamp: video zamanı (saniye).
        Eşik geçildiyse True döner, geçilmediyse False.
        """
        if closed_detected:
            if self._closed_since is None:
                self._closed_since = timestamp
            return (timestamp - self._closed_since) >= self.threshold
        else:
            self._closed_since = None
            return False

    def reset(self):
        self._closed_since = None


_drowsiness_tracker = _DrowsinessTracker()

# ── ROI / geometry helpers ────────────────────────────────────────────────────

def _roi_from_fracs(frame, car_bbox, fracs):
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if car_bbox is not None:
        cx1, cy1, cx2, cy2 = car_bbox
        cw, ch = cx2 - cx1, cy2 - cy1
    else:
        cx1, cy1, cw, ch = 0, 0, w, h
    fx1, fy1, fx2, fy2 = fracs
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


def _bbox_contains(outer, inner, overlap=0.5):
    """True if inner overlaps outer by at least `overlap` fraction of inner's area."""
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    inter_x = max(0.0, min(ox2, ix2) - max(ox1, ix1))
    inter_y = max(0.0, min(oy2, iy2) - max(oy1, iy1))
    inter_area = inter_x * inter_y
    inner_area = max(1.0, (ix2 - ix1) * (iy2 - iy1))
    return (inter_area / inner_area) >= overlap


def _select_driver_person(object_detections, roi=None):
    if not object_detections:
        return None
    candidates = [
        d for d in object_detections
        if utils.to_ascii(d.get("label")) in PERSON_LABELS
        and "bbox" in d
        and (roi is None or _bbox_intersects_roi(d["bbox"], roi))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: _bbox_center_x(d["bbox"]))

# ── EAR analysis ──────────────────────────────────────────────────────────────

def _ear_analyze(dets):
    """Returns (actions, closed_eye_conf).
    actions: [{"label", "conf"}, ...] for esneme.
    closed_eye_conf: max confidence of closed_eye detections (0.0 if none).
    """
    faces = [d for d in dets if d["label"] == "face"]
    features = [d for d in dets if d["label"] != "face"]
    if faces:
        anchor = max(faces, key=lambda d: d["confidence"])
        features = [d for d in features if _bbox_contains(anchor["bbox"], d["bbox"])]

    out = []
    open_mouths = [d for d in features if d["label"] == "open_mouth"]
    if open_mouths:
        out.append({"label": "esneme", "conf": max(d["confidence"] for d in open_mouths)})

    closed_eyes = [d for d in features if d["label"] in {"closed_eye", "closed_eyes"}]
    closed_conf = max((d["confidence"] for d in closed_eyes), default=0.0)
    return out, closed_conf

# ── public API ────────────────────────────────────────────────────────────────

def load_model(models_dir):
    return {
        "action": utils.load_yolo(os.path.join(models_dir, ACTION_WEIGHT_FILE)),
        "ear":    utils.load_yolo(os.path.join(models_dir, EAR_WEIGHT_FILE)),
    }


def detect(models, frame, device, object_detections=None, car_bbox=None, timestamp=0.0):
    """
    Detect driver actions and EAR-based states (esneme, uykululuk).
    `models` must be the dict returned by load_model().
    `timestamp`: video zamanı (saniye) — uykululuk için zaman takibinde kullanılır.
    Returns [{"label": str, "conf": float}, ...].
    uykululuk, formatter'da yarışma şemasına göre filtrelenir.
    """
    if not models or frame is None:
        return []

    out = []

    # --- action detection ---
    action_model = models.get("action")
    if action_model is not None:
        try:
            roi = _roi_from_fracs(frame, car_bbox, _DRIVER_ROI_FRACS)
            driver_crop = None
            if object_detections is not None:
                selected = _select_driver_person(object_detections, roi=roi)
                if selected is not None:
                    driver_crop = utils.crop_bbox(frame, selected["bbox"], pad=0.10)
            if driver_crop is None and roi is not None:
                x1, y1, x2, y2 = roi
                driver_crop = frame[y1:y2, x1:x2]
            if driver_crop is not None and driver_crop.size > 0:
                result = action_model(driver_crop, conf=CONF_THRESHOLD, device=device, verbose=False)[0]
                dets = utils.detections_from_result(result, conf_threshold=CONF_THRESHOLD)
                kemer_takili_detected = False
                for d in dets:
                    raw = utils.to_ascii(d["label"])
                    if raw == "kemer_takili":
                        kemer_takili_detected = True
                        continue  # kemer takili → ihlal yok, çıktıya ekleme
                    mapped = LABEL_MAP.get(raw, raw)
                    if mapped in VALID_ACTIONS:
                        out.append({"label": mapped, "conf": d["confidence"]})
                if not kemer_takili_detected:
                    # Sürücü bölgesi görüldü ama kemer tespit edilemedi → ihlal
                    out.append({"label": "emniyet_kemeri_ihlali", "conf": 0.80})
        except Exception:
            pass

    # --- EAR detection ---
    ear_model = models.get("ear")
    if ear_model is not None:
        try:
            roi = _roi_from_fracs(frame, car_bbox, _FACE_ROI_FRACS)
            if roi is not None:
                rx1, ry1, rx2, ry2 = roi
                roi_frame = frame[ry1:ry2, rx1:rx2]
                if roi_frame is not None and roi_frame.size > 0:
                    result = ear_model(roi_frame, conf=CONF_THRESHOLD, device=device, verbose=False)[0]
                    dets = utils.detections_from_result(result, conf_threshold=CONF_THRESHOLD)
                    ear_actions, closed_conf = _ear_analyze(dets)
                    out.extend(ear_actions)
                    # Uykululuk: gözler CLOSED_EYE_THRESHOLD saniyeden uzun kapalıysa
                    if _drowsiness_tracker.update(closed_conf > 0.0, timestamp):
                        out.append({"label": "uykululuk", "conf": closed_conf})
                else:
                    _drowsiness_tracker.update(False, timestamp)
        except Exception:
            _drowsiness_tracker.update(False, timestamp)

    return out
