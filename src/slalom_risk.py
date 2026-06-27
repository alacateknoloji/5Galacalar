#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vehicle slalom and speed risk detection module."""

import glob
import json
import math
import os
import re

try:
    from src import utils, vehicle_type
except ModuleNotFoundError:
    import utils
    import vehicle_type

# ── Kullanıcı Ayarları ─────────────────────────────────────────────────
MODEL_PATH = None
SOURCE = None
VIDEO_FPS = 30.0
CONF_THRESHOLD = 0.25
LABEL_MAP = {}

MAX_HISTORY_SECONDS = 2
MIN_LATERAL_MOVEMENT = 30    # Anlamlı yatay hareket eşiği (piksel)
ZIGZAG_THRESHOLD = 2         # Slalom için gereken min yön değişimi
MAX_TRACK_DISTANCE = 100     # Eşleştirme için max merkez mesafesi (piksel)

PIXELS_PER_METER = 10.0      # Kamerana göre kalibre et!
SPEED_LIMIT_KMH = 50.0
SPEED_SMOOTH_FRAMES = 10

SHOW_VIDEO = True

VEHICLE_LABELS = ["ambulans", "hatchback", "itfaiye", "kamyon", "minibus",
                  "panelvan", "pickup", "sedan", "suv"]


# ── Yardımcılar ────────────────────────────────────────────────────────
def resolve_model_path(models_dir=None):
    candidates = [models_dir, os.environ.get("MODELS_DIR"),
                  os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), "models"),
                  "/app/models"]
    for candidate in candidates:
        if not candidate or not os.path.isdir(candidate):
            continue
        for path in utils._candidate_weight_paths(os.path.join(candidate, vehicle_type.WEIGHT_FILE)):
            if os.path.isfile(path):
                return path
    return None


def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_distance(c1, c2):
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

    def _max_history_frames(self, fps):
        return max(2, int(round(self.max_history_seconds * max(fps, 1))))

    def _create_track(self, det, frame_index):
        tid = self.next_id
        self.next_id += 1
        self.tracks[tid] = {
            "track_id": tid,
            "label": det["label"],
            "bbox": det["bbox"],
            "confidence": det["confidence"],
            "center": det["center"],
            "last_frame": frame_index,
            "history": [(frame_index, det["center"][0], det["center"][1])],
        }
        return tid

    def update(self, detections, frame_index, fps):
        dets = [
            {"label": d["label"], "bbox": d["bbox"],
             "confidence": float(d.get("confidence", 0.0)),
             "center": get_center(d["bbox"])}
            for d in detections if d.get("label") in self.vehicle_labels
        ]

        pairs = []
        for di, det in enumerate(dets):
            for tid in self.tracks:
                dist = center_distance(det["center"], self.tracks[tid]["center"])
                if dist <= self.max_distance:
                    pairs.append((dist, di, tid))
        pairs.sort(key=lambda p: p[0])

        assignment, matched_dets, matched_tracks = {}, set(), set()
        for _, di, tid in pairs:
            if di in matched_dets or tid in matched_tracks:
                continue
            assignment[di] = tid
            matched_dets.add(di)
            matched_tracks.add(tid)

        updated_ids = []
        for di, det in enumerate(dets):
            if di in assignment:
                tid = assignment[di]
                track = self.tracks[tid]
                track.update({"label": det["label"], "bbox": det["bbox"],
                              "confidence": det["confidence"], "center": det["center"],
                              "last_frame": frame_index})
                track["history"].append((frame_index, det["center"][0], det["center"][1]))
            else:
                tid = self._create_track(det, frame_index)
            updated_ids.append(tid)

        self._trim_history(frame_index, fps)
        self._prune_tracks(frame_index, fps)
        return [self.tracks[tid] for tid in updated_ids]

    def _trim_history(self, frame_index, fps):
        min_frame = frame_index - self._max_history_frames(fps)
        for track in self.tracks.values():
            track["history"] = [h for h in track["history"] if h[0] >= min_frame]

    def _prune_tracks(self, frame_index, fps):
        limit = self._max_history_frames(fps)
        stale = [tid for tid, track in self.tracks.items() if frame_index - track["last_frame"] > limit]
        for tid in stale:
            del self.tracks[tid]


# ── Slalom Analizi ─────────────────────────────────────────────────────
class SlalomRiskAnalyzer:
    def __init__(self, min_lateral=MIN_LATERAL_MOVEMENT, zigzag_threshold=ZIGZAG_THRESHOLD):
        self.min_lateral = min_lateral
        self.zigzag_threshold = zigzag_threshold

    def _count_direction_changes(self, xs):
        if len(xs) < 2:
            return 0
        sig_dirs, last_extreme, move_dir = [], xs[0], 0
        for x in xs[1:]:
            diff = x - last_extreme
            if move_dir == 0:
                if abs(diff) >= self.min_lateral:
                    move_dir = 1 if diff > 0 else -1
                    sig_dirs.append(move_dir)
                    last_extreme = x
            elif (diff > 0 and move_dir > 0) or (diff < 0 and move_dir < 0):
                last_extreme = x
            elif abs(diff) >= self.min_lateral:
                move_dir = 1 if diff > 0 else -1
                sig_dirs.append(move_dir)
                last_extreme = x
        return sum(1 for a, b in zip(sig_dirs, sig_dirs[1:]) if a != b)

    def analyze(self, track):
        zigzag_count = self._count_direction_changes([h[1] for h in track["history"]])
        slalom = zigzag_count >= self.zigzag_threshold
        confidence, reason = 0.0, ""
        if slalom:
            extra = zigzag_count - self.zigzag_threshold
            confidence = round(min(0.99, track["confidence"] * (0.90 + 0.03 * extra)), 2)
            reason = "Araç kısa süre içinde sağ-sol yön değişimi yaptı."
        return {"slalom": slalom, "zigzag_count": zigzag_count, "confidence": confidence, "reason": reason}


# ── Hız Tahmini ────────────────────────────────────────────────────────
class SpeedEstimator:
    def estimate(self, track, fps):
        history = track["history"]
        if len(history) < 2:
            return 0.0
        recent = history[-SPEED_SMOOTH_FRAMES:]
        if len(recent) < 2:
            return 0.0
        total_px = sum(math.hypot(recent[i][1] - recent[i - 1][1], recent[i][2] - recent[i - 1][2])
                       for i in range(1, len(recent)))
        frames_elapsed = recent[-1][0] - recent[0][0]
        if frames_elapsed == 0:
            return 0.0
        return round((total_px / frames_elapsed) * fps / PIXELS_PER_METER * 3.6, 1)


# ── Ana Sistem ─────────────────────────────────────────────────────────
class SlalomDetectionSystem:
    def __init__(self, max_distance=MAX_TRACK_DISTANCE, min_lateral=MIN_LATERAL_MOVEMENT,
                 zigzag_threshold=ZIGZAG_THRESHOLD, vehicle_labels=None):
        self.tracker = VehicleTracker(max_distance=max_distance, vehicle_labels=vehicle_labels)
        self.analyzer = SlalomRiskAnalyzer(min_lateral=min_lateral, zigzag_threshold=zigzag_threshold)
        self.speed_estimator = SpeedEstimator()

    def process_frame(self, detections, frame_index, fps):
        tracks = self.tracker.update(detections, frame_index, fps)
        slalom_detections, all_tracks = [], []
        for track in tracks:
            res = self.analyzer.analyze(track)
            speed_kmh = self.speed_estimator.estimate(track, fps)
            speeding = speed_kmh > SPEED_LIMIT_KMH
            entry = {
                "track_id": track["track_id"],
                "label": track["label"],
                "bbox": [int(round(v)) for v in track["bbox"]],
                "center": [int(round(track["center"][0])), int(round(track["center"][1]))],
                "speed_kmh": speed_kmh,
                "speeding": speeding,
                "slalom": res["slalom"],
                "confidence": round(track["confidence"], 2),
            }
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
    if boxes is None or len(boxes) == 0:
        return []
    names = result.names
    dets = []
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        raw = names[cls_id] if isinstance(names, (list, tuple)) else names.get(cls_id, str(cls_id))
        label = label_map.get(raw, raw) if label_map else raw
        x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i].tolist())
        dets.append({"label": label, "bbox": [x1, y1, x2, y2], "confidence": float(boxes.conf[i].item())})
    return dets


def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _collect_sources(source):
    if not source:
        return [], None
    if os.path.isdir(source):
        exts = ("jpg", "jpeg", "png", "bmp", "webp")
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
        alert = t.get("slalom", False) or t.get("speeding", False)
        color = (0, 0, 255) if alert else (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{t['label']} #{t['track_id']} | {t['speed_kmh']} km/h",
                    (x1, max(y1 - 22, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        warnings = []
        if t.get("slalom"):
            warnings.append("SLALOM!")
        if t.get("speeding"):
            warnings.append(f"HIZ ASIMI! (>{SPEED_LIMIT_KMH:.0f}km/h)")
        if warnings:
            cv2.putText(frame, " | ".join(warnings), (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return frame


def _print_frame_info(out, frame_index):
    for t in out.get("all_tracks", []):
        flags = []
        if t.get("slalom"):
            flags.append("SLALOM")
        if t.get("speeding"):
            flags.append(f"HIZ ASIMI (>{SPEED_LIMIT_KMH:.0f}km/h)")
        flag_str = "  *** " + " | ".join(flags) + " ***" if flags else ""
        print(f"[KARE {frame_index}] #{t['track_id']} {t['label']:12s} {t['speed_kmh']:>6.1f} km/h{flag_str}")


def run_inference(model_path=None, source=SOURCE, fps=VIDEO_FPS, conf=CONF_THRESHOLD, label_map=None):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Bu adim icin 'ultralytics' gerekli.\nKur: pip install ultralytics") from exc

    cv2_ok = False
    if SHOW_VIDEO:
        try:
            import cv2
            cv2_ok = True
        except ImportError:
            print("UYARI: OpenCV bulunamadi; gorsel pencere acilmayacak.")

    if model_path is None:
        model_path = resolve_model_path()
    if not model_path or not os.path.exists(model_path):
        raise SystemExit(f"Model bulunamadi: {model_path}\nMODELS_DIR veya MODEL_PATH'i kontrol et.")

    model = YOLO(model_path)
    reset_system()
    items, kind = _collect_sources(source)
    if not items:
        raise SystemExit(f"Kaynak bos/bulunamadi: {source}")
    if kind == "images" and len(items) == 1:
        print("UYARI: Tek goruntu - slalom zamansal bir davranistir, sirali kareler kullan.")

    outputs = []
    source_iter = items if kind == "images" else model(source, stream=True, conf=conf, verbose=False)
    for frame_index, item in enumerate(source_iter):
        result = model(item, conf=conf, verbose=False)[0] if kind == "images" else item
        out = process_frame(detections_from_result(result, label_map), frame_index, fps)
        outputs.append(out)
        _print_frame_info(out, frame_index)
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
    outputs = run_inference(model_path=MODEL_PATH, source=SOURCE, fps=VIDEO_FPS,
                            conf=CONF_THRESHOLD, label_map=LABEL_MAP or None)
    slalom_hits = sum(1 for o in outputs if o["slalom_detections"])
    speed_hits = sum(1 for o in outputs if any(t["speeding"] for t in o.get("all_tracks", [])))
    print(f"\n[OZET] Toplam {len(outputs)} kare islendi.")
    print(f"  Slalom   : {slalom_hits} karede tespit edildi.")
    print(f"  Hiz asimi: {speed_hits} karede tespit edildi (esik: {SPEED_LIMIT_KMH:.0f} km/h).")
    print(f"  Not: PIXELS_PER_METER={PIXELS_PER_METER} — kamerana gore kalibre et!")
    if not outputs:
        print("Hic kare islenemedi. SOURCE yolunu kontrol et.")
