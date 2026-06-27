# -*- coding: utf-8 -*-
"""
Inference orchestrator.

run_inference(video_path) reads the video frame by frame and runs every module,
then hands the aggregated raw results to the formatter for schema-exact output.

Cadence (tune for the 10-minute limit):
  - vehicle detection + slalom run at ANALYZE_FPS (needs temporal continuity).
  - colour / plate / driver / object / passenger run at HEAVY_FPS (per-instant
    classifications that do not need a high frame rate). This keeps the run well
    inside the time budget on a Tesla T4.

There is no wall-clock branch and no environment check here: the same frames are
processed the same way regardless of where the code runs.
"""

import os
from collections import Counter

import cv2

try:
    from src import (
        utils,
        vehicle_type,
        vehicle_color,
        plate_ocr,
        driver_behavior,
        object_detection,
        passenger_detection,
        formatter,
    )
    from src.slalom_risk import SlalomDetectionSystem
except ModuleNotFoundError:
    import utils
    import vehicle_type
    import vehicle_color
    import plate_ocr
    import driver_behavior
    import object_detection
    import passenger_detection
    import formatter
    from slalom_risk import SlalomDetectionSystem

# >>> CONFIG: where the .pt weights live (matches the VM spec table) <<<
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _resolve_models_dir(models_dir=None):
    candidates = []
    if models_dir:
        candidates.append(models_dir)
    candidates.extend([
        os.environ.get("MODELS_DIR"),
        os.path.join(ROOT_DIR, "models"),
        "/app/models",
    ])
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return os.path.join(ROOT_DIR, "models")


MODELS_DIR = _resolve_models_dir()

# Analysis frame rates.
ANALYZE_FPS = 5.0   # vehicle + slalom
HEAVY_FPS = 2.0     # colour / plate / driver / object / passenger

# Base tracker match distance at full frame rate; scaled up by the stride below
# because a sub-sampled vehicle travels further between processed frames.
BASE_TRACK_DISTANCE = 100


def run_inference(video_path, models_dir=None):
    models_dir = _resolve_models_dir(models_dir)
    video_id = os.path.basename(video_path)
    device = utils.get_device()
    utils.log.info("device=%s, video=%s", device, video_id)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video could not be opened: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps != fps:  # 0 / None / NaN guard
        fps = 30.0
        utils.log.warning("FPS unreadable; assuming %.1f", fps)

    stride = max(1, int(round(fps / ANALYZE_FPS)))          # vehicle/slalom cadence
    heavy_every = max(1, int(round(ANALYZE_FPS / HEAVY_FPS)))  # in analysed-frame units

    # Load every model up-front (each returns None safely if its weight is absent).
    veh_model = vehicle_type.load_model(models_dir)
    color_model = vehicle_color.load_model(models_dir)
    plate_model = plate_ocr.load_model(models_dir)
    driver_model = driver_behavior.load_model(models_dir)
    object_model = object_detection.load_model(models_dir)
    passenger_model = passenger_detection.load_model(models_dir)

    slalom_system = SlalomDetectionSystem(max_distance=BASE_TRACK_DISTANCE * stride)

    type_votes, color_votes, plate_votes = Counter(), Counter(), Counter()
    veh_confs, color_confs, plate_confs = [], [], []
    tespitler = []

    frame_idx = -1
    analyzed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % stride != 0:
            continue
        analyzed += 1
        t = round(frame_idx / fps, 1)

        # --- vehicle detection (shared by type + slalom) ---
        try:
            dets = vehicle_type.detect(veh_model, frame, device)
        except Exception:
            dets = []

        # --- slalom + hız (temporal) ---
        try:
            sres = slalom_system.process_frame(dets, frame_idx, fps)
            for sd in sres["slalom_detections"]:
                detection = {
                    "zaman_saniye": t,
                    "kategori": "sofor_eylemi",
                    "etiket": "slalom",
                    "confidence_score": sd["confidence"],
                }
                if "speed_kmh" in sd:
                    detection["speed_kmh"] = sd["speed_kmh"]
                tespitler.append(detection)
            for tr in sres["all_tracks"]:
                if tr.get("speeding"):
                    tespitler.append({
                        "zaman_saniye": t,
                        "kategori": "sofor_eylemi",
                        "etiket": "hiz_asimi",
                        "confidence_score": tr.get("confidence", 0.5),
                        "speed_kmh": tr["speed_kmh"],
                    })
        except Exception:
            pass

        # --- main vehicle type ---
        tip, tconf, main_bbox = vehicle_type.classify_main(dets)
        if tip:
            type_votes[tip] += 1
            veh_confs.append(tconf)

        run_heavy = (analyzed % heavy_every == 0)

        # --- colour + plate (need the main vehicle box) ---
        if run_heavy and main_bbox is not None:
            try:
                crop = utils.crop_bbox(frame, main_bbox, pad=0.02)
                color, cconf = vehicle_color.classify(color_model, crop, device)
                if color:
                    color_votes[color] += 1
                    color_confs.append(cconf)
            except Exception:
                pass
            try:
                plate, pconf = plate_ocr.read_plate(plate_model, frame, main_bbox, device)
                if plate and plate != utils.PLATE_NOT_DETECTED:
                    plate_votes[plate] += 1
                    plate_confs.append(pconf)
            except Exception:
                pass

        # --- driver / object / passenger ---
        if run_heavy:
            object_detections = object_detection.detect(object_model, frame, device, main_bbox)
            for a in driver_behavior.detect(driver_model, frame, device, object_detections, main_bbox):
                tespitler.append({"zaman_saniye": t, "kategori": "sofor_eylemi",
                                   "etiket": a["label"], "confidence_score": a["conf"]})
            for o in object_detections:
                tespitler.append({"zaman_saniye": t, "kategori": "nesneler",
                                  "etiket": o["label"], "confidence_score": o["conf"]})
            for p in passenger_detection.detect(passenger_model, frame, device, object_detections, main_bbox):
                tespitler.append({"zaman_saniye": t, "kategori": "yolcular",
                                  "etiket": p["label"], "confidence_score": p["conf"]})

    cap.release()

    # --- aggregate arac_bilgisi (majority vote) ---
    tip = type_votes.most_common(1)[0][0] if type_votes else None
    renk = color_votes.most_common(1)[0][0] if color_votes else None
    plaka = plate_votes.most_common(1)[0][0] if plate_votes else utils.PLATE_NOT_DETECTED
    arac_conf = _combine_confidence(veh_confs, color_confs, plate_confs)

    raw = {
        "tip": tip,
        "renk": renk,
        "plaka": plaka,
        "arac_confidence": arac_conf,
        "tespitler": tespitler,
    }
    result = formatter.format_output(raw, video_id)
    utils.log.info("done: %d frames analysed, %d detections kept",
                   analyzed, len(result["tespitler"]))
    return result


def _combine_confidence(*conf_lists):
    """Average the per-attribute mean confidences into one vehicle score."""
    means = []
    for lst in conf_lists:
        if lst:
            means.append(sum(lst) / len(lst))
    if not means:
        return 0.0
    return sum(means) / len(means)
