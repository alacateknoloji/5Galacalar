#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import glob
import json
import math

try:
    from src import utils, vehicle_type
except ModuleNotFoundError:
    import utils
    import vehicle_type

# ── Kullanıcı Ayarları ─────────────────────────────────────────────────
MODEL_PATH = None   # None → resolve_model_path() ile otomatik bulunur
SOURCE     = None   # Sıralı görüntü klasörü veya video dosyası

VIDEO_FPS            = 30.0
CONF_THRESHOLD       = 0.25
LABEL_MAP            = {}

MAX_HISTORY_SECONDS  = 2
MIN_LATERAL_MOVEMENT = 30    # Anlamlı yatay hareket eşiği (piksel)
ZIGZAG_THRESHOLD     = 2     # Slalom için gereken min yön değişimi
MAX_TRACK_DISTANCE   = 100   # Eşleştirme için max merkez mesafesi (piksel)

PIXELS_PER_METER     = 10.0  # Kamerana göre kalibre et!
SPEED_LIMIT_KMH      = 50.0
SPEED_SMOOTH_FRAMES  = 10

SHOW_VIDEO = True

VEHICLE_LABELS = ["ambulans", "hatchback", "itfaiye", "kamyon", "minibus",
                   "panelvan", "pickup", "sedan", "suv"]


# ── Yardımcılar ────────────────────────────────────────────────────────
def resolve_model_path(models_dir=None):
    candidates = [models_dir, os.environ.get("MODELS_DIR"),
                  os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), "models"),
                  "/app/models"]
    for d in candidates:
        if not d or not os.path.isdir(d):
            continue
        for p in utils._candidate_weight_paths(os.path.join(d, vehicle_type.WEIGHT_FILE)):
            if os.path.isfile(p):
                return p
    return None


def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _dist(c1, c2):
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


# ── Araç Takibi ────────────────────────────────────────────────────────
class VehicleTracker:
    def __init__(self, max_distance=MAX_TRACK_DISTANCE, max_history_seconds=MAX_HISTORY_SECONDS,
                 vehicle_labels=None):
        self.max_distance = max_distance
        self.max_history_seconds = max_history_seconds
        self.vehicle_labels = set(vehicle_labels or VEHICLE_LABELS)
        self.tracks = {}
        self.next_id = 1

    def _history_frames(self, fps):
        return max(2, int(round(self.max_history_seconds * max(fps, 1))))

    def update(self, detections, frame_index, fps):
        dets = [{"label": d["label"], "bbox": d["bbox"],
                 "confidence": float(d.get("confidence", 0)),
                 "center": get_center(d["bbox"])}
                for d in detections if d.get("label") in self.vehicle_labels]

        # Greedy en-yakın eşleştirme
        pairs = sorted(
            [(d_i, tid, _dist(d["center"], self.tracks[tid]["center"]))
             for d_i, d in enumerate(dets)
             for tid in self.tracks
             if _dist(d["center"], self.tracks[tid]["center"]) <= self.max_distance],
            key=lambda x: x[2])
        matched_d, matched_t, assignment = set(), set(), {}
        for d_i, tid, _ in pairs:
            if d_i not in matched_d and tid not in matched_t:
                matched_d.add(d_i); matched_t.add(tid); assignment[d_i] = tid

        updated = []
        for d_i, d in enumerate(dets):
            if d_i in assignment:
                t = self.tracks[assignment[d_i]]
                t.update({"label": d["label"], "bbox": d["bbox"],
                           "confidence": d["confidence"], "center": d["center"],
                           "last_frame": frame_index})
                t["history"].append((frame_index, *d["center"]))
                updated.append(assignment[d_i])
            else:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {"track_id": tid, "label": d["label"], "bbox": d["bbox"],
                                     "confidence": d["confidence"], "center": d["center"],
                                     "last_frame": frame_index,
                                     "history": [(frame_index, *d["center"])]}
                updated.append(tid)

        # Geçmişi kırp + eski track'leri temizle
        limit = self._history_frames(fps)
        min_frame = frame_index - limit
        stale = []
        for tid, t in self.tracks.items():
            t["history"] = [h for h in t["history"] if h[0] >= min_frame]
            if frame_index - t["last_frame"] > limit:
                stale.append(tid)
        for tid in stale:
            del self.tracks[tid]

        return [self.tracks[tid] for tid in updated]


# ── Slalom Analizi ─────────────────────────────────────────────────────
class SlalomRiskAnalyzer:
    def __init__(self, min_lateral=MIN_LATERAL_MOVEMENT, zigzag_threshold=ZIGZAG_THRESHOLD):
        self.min_lateral = min_lateral
        self.zigzag_threshold = zigzag_threshold

    def _direction_changes(self, xs):
        if len(xs) < 2:
            return 0
        dirs, last_ext, cur_dir = [], xs[0], 0
        for x in xs[1:]:
            diff = x - last_ext
            if cur_dir == 0:
                if abs(diff) >= self.min_lateral:
                    cur_dir = 1 if diff > 0 else -1
                    dirs.append(cur_dir); last_ext = x
            else:
                same = (diff > 0 and cur_dir > 0) or (diff < 0 and cur_dir < 0)
                if same:
                    last_ext = x
                elif abs(diff) >= self.min_lateral:
                    cur_dir = 1 if diff > 0 else -1
                    dirs.append(cur_dir); last_ext = x
        return sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1])

    def analyze(self, track):
        count  = self._direction_changes([h[1] for h in track["history"]])
        slalom = count >= self.zigzag_threshold
        confidence, reason = 0.0, ""
        if slalom:
            confidence = round(min(0.99, track["confidence"] * (0.90 + 0.03 * (count - self.zigzag_threshold))), 2)
            reason = "Araç kısa süre içinde sağ-sol yön değişimi yaptı."
        return {"slalom": slalom, "zigzag_count": count, "confidence": confidence, "reason": reason}


# ── Hız Tahmini ────────────────────────────────────────────────────────
def estimate_speed(track, fps):
    recent = track["history"][-SPEED_SMOOTH_FRAMES:]
    if len(recent) < 2:
        return 0.0
    total_px = sum(math.hypot(recent[i][1] - recent[i-1][1], recent[i][2] - recent[i-1][2])
                   for i in range(1, len(recent)))
    elapsed = recent[-1][0] - recent[0][0]
    return round(total_px / elapsed * fps / PIXELS_PER_METER * 3.6, 1) if elapsed else 0.0


# ── Ana Sistem ─────────────────────────────────────────────────────────
class SlalomDetectionSystem:
    def __init__(self, max_distance=MAX_TRACK_DISTANCE, min_lateral=MIN_LATERAL_MOVEMENT,
                 zigzag_threshold=ZIGZAG_THRESHOLD, vehicle_labels=None):
        self.tracker  = VehicleTracker(max_distance=max_distance, vehicle_labels=vehicle_labels)
        self.analyzer = SlalomRiskAnalyzer(min_lateral=min_lateral, zigzag_threshold=zigzag_threshold)

    def process_frame(self, detections, frame_index, fps):
        all_tracks, slalom_detections = [], []
        for t in self.tracker.update(detections, frame_index, fps):
            res      = self.analyzer.analyze(t)
            speed    = estimate_speed(t, fps)
            speeding = speed > SPEED_LIMIT_KMH
            entry    = {"track_id": t["track_id"], "label": t["label"],
                        "bbox": [int(round(v)) for v in t["bbox"]],
                        "center": [int(round(v)) for v in t["center"]],
                        "speed_kmh": speed, "speeding": speeding, "slalom": res["slalom"]}
            all_tracks.append(entry)
            if res["slalom"]:
                slalom_detections.append({**entry, "reason": res["reason"], "confidence": res["confidence"]})
        return {"frame_index": frame_index, "slalom_detections": slalom_detections, "all_tracks": all_tracks}


_default_system = SlalomDetectionSystem()


def process_frame(detections, frame_index, fps):
    """detections: [{"label": str, "bbox": [x1,y1,x2,y2], "confidence": float}, ...]"""
    return _default_system.process_frame(detections, frame_index, fps)


def reset_system():
    global _default_system
    _default_system = SlalomDetectionSystem()


# ── YOLO Çıkarım Katmanı ───────────────────────────────────────────────
def detections_from_result(result, label_map=None):
    boxes = getattr(result, "boxes", None)
    if not boxes or len(boxes) == 0:
        return []
    names = result.names
    dets  = []
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        raw    = names[cls_id] if isinstance(names, (list, tuple)) else names.get(cls_id, str(cls_id))
        label  = (label_map or {}).get(raw, raw)
        x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i].tolist())
        dets.append({"label": label, "bbox": [x1, y1, x2, y2], "confidence": float(boxes.conf[i].item())})
    return dets


def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _collect_sources(source):
    if os.path.isdir(source):
        exts  = ("jpg", "jpeg", "png", "bmp", "webp")
        files = []
        for e in exts:
            files += glob.glob(os.path.join(source, f"*.{e}")) + glob.glob(os.path.join(source, f"*.{e.upper()}"))
        return sorted(set(files), key=_natural_key), "images"
    return [source], "video"


def _draw_annotations(frame, out):
    try:
        import cv2
    except ImportError:
        return frame
    for t in out.get("all_tracks", []):
        x1, y1, x2, y2 = t["bbox"]
        alert = t.get("slalom") or t.get("speeding")
        color = (0, 0, 255) if alert else (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{t['label']} #{t['track_id']} | {t['speed_kmh']} km/h",
                    (x1, max(y1 - 22, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        warnings = (["SLALOM!"] if t.get("slalom") else []) + \
                   ([f"HIZ ASIMI! (>{SPEED_LIMIT_KMH:.0f}km/h)"] if t.get("speeding") else [])
        if warnings:
            cv2.putText(frame, " | ".join(warnings), (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return frame


def run_inference(model_path=None, source=SOURCE, fps=VIDEO_FPS, conf=CONF_THRESHOLD, label_map=None):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Bu adim icin 'ultralytics' gerekli.\nKur: pip install ultralytics")

    cv2_ok = False
    if SHOW_VIDEO:
        try:
            import cv2; cv2_ok = True
        except ImportError:
            print("UYARI: OpenCV bulunamadi; gorsel pencere acilmayacak.")

    if model_path is None:
        model_path = resolve_model_path()
    if not model_path or not os.path.exists(model_path):
        raise SystemExit(f"Model bulunamadi: {model_path}")

    model = YOLO(model_path)
    reset_system()
    items, kind = _collect_sources(source)
    if not items:
        raise SystemExit(f"Kaynak bos/bulunamadi: {source}")
    if kind == "images" and len(items) == 1:
        print("UYARI: Tek goruntu - slalom zamansal bir davranistir, sirali kareler kullan.")

    outputs      = []
    source_iter  = items if kind == "images" else model(source, stream=True, conf=conf, verbose=False)

    for frame_index, item in enumerate(source_iter):
        result = model(item, conf=conf, verbose=False)[0] if kind == "images" else item
        out    = process_frame(detections_from_result(result, label_map), frame_index, fps)
        outputs.append(out)

        for t in out.get("all_tracks", []):
            flags    = (["SLALOM"] if t.get("slalom") else []) + \
                       ([f"HIZ ASIMI (>{SPEED_LIMIT_KMH:.0f}km/h)"] if t.get("speeding") else [])
            flag_str = "  *** " + " | ".join(flags) + " ***" if flags else ""
            print(f"[KARE {frame_index}] #{t['track_id']} {t['label']:12s} {t['speed_kmh']:>6.1f} km/h{flag_str}")

        if out["slalom_detections"]:
            print(json.dumps(out["slalom_detections"], ensure_ascii=False))

        if cv2_ok and SHOW_VIDEO:
            import cv2
            cv2.imshow("Slalom + Hiz Takibi", _draw_annotations(result.orig_img.copy(), out))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if cv2_ok and SHOW_VIDEO:
        import cv2
        cv2.destroyAllWindows()

    return outputs


# ── Çalıştırma ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    outputs  = run_inference(MODEL_PATH, SOURCE, VIDEO_FPS, CONF_THRESHOLD, LABEL_MAP or None)
    slalom_n = sum(1 for o in outputs if o["slalom_detections"])
    speed_n  = sum(1 for o in outputs if any(t["speeding"] for t in o.get("all_tracks", [])))
    print(f"\n[OZET] {len(outputs)} kare islendi.")
    print(f"  Slalom   : {slalom_n} karede tespit edildi.")
    print(f"  Hiz asimi: {speed_n} karede (esik: {SPEED_LIMIT_KMH:.0f} km/h).")
    if not outputs:
        print("Hic kare islenemedi. SOURCE yolunu kontrol et.")
