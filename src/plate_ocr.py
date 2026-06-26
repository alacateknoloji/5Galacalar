# -*- coding: utf-8 -*-
"""
License plate module.

Pipeline:  detect plate box (plate.pt)  ->  OCR the crop  ->  normalise to the
official Turkish plate regex.  Returns utils.PLATE_NOT_DETECTED when no valid
plate can be produced.

OCR backend: EasyOCR by default. A PaddleOCR alternative is provided below;
switch OCR_BACKEND to "paddle" to use it. The OCR reader is created lazily and
cached so it is built only once per run.
"""

import os

try:
    from src import utils
except ModuleNotFoundError:
    import utils

# >>> CONFIG: plate-detection weight file inside the models dir <<<
WEIGHT_FILE = "plate.pt"

# "easyocr" or "paddle"
OCR_BACKEND = "easyocr"

PLATE_CONF_THRESHOLD = 0.25

_reader = None  # cached OCR reader


def load_model(models_dir):
    return utils.load_yolo(os.path.join(models_dir, WEIGHT_FILE))


def _get_reader():
    """Build (once) and return the OCR reader, or None if OCR is unavailable."""
    global _reader
    if _reader is not None:
        return _reader
    device = utils.get_device()
    gpu = device != "cpu"
    try:
        if OCR_BACKEND == "paddle":
            from paddleocr import PaddleOCR
            _reader = ("paddle", PaddleOCR(use_angle_cls=True, lang="en", show_log=False, use_gpu=gpu))
        else:
            import easyocr
            _reader = ("easyocr", easyocr.Reader(["en"], gpu=gpu))
        return _reader
    except Exception as e:  # noqa: BLE001
        utils.log.warning("OCR backend '%s' unavailable (%s); plates disabled", OCR_BACKEND, e)
        _reader = ("none", None)
        return _reader


def _ocr_text(crop):
    """Run OCR on a plate crop and return the raw concatenated text."""
    kind, reader = _get_reader()
    if reader is None:
        return ""
    try:
        if kind == "paddle":
            out = reader.ocr(crop, cls=True)
            if not out or not out[0]:
                return ""
            return "".join(line[1][0] for line in out[0])
        else:  # easyocr
            out = reader.readtext(crop, detail=0)
            return "".join(out) if out else ""
    except Exception:
        return ""


def read_plate(model, frame, main_bbox, device):
    """
    Detect the plate within the main vehicle bbox, OCR it and normalise.
    Returns (plate_string, confidence). plate_string is utils.PLATE_NOT_DETECTED
    when nothing valid is found.
    """
    if model is None or frame is None:
        return utils.PLATE_NOT_DETECTED, 0.0
    try:
        roi_frame = frame
        if main_bbox is not None:
            roi_frame = utils.crop_bbox(frame, main_bbox, pad=0.02)
        if roi_frame is None or roi_frame.size == 0:
            return utils.PLATE_NOT_DETECTED, 0.0

        result = model(roi_frame, conf=PLATE_CONF_THRESHOLD, device=device, verbose=False)[0]
        dets = utils.detections_from_result(result, conf_threshold=PLATE_CONF_THRESHOLD)
        if not dets:
            return utils.PLATE_NOT_DETECTED, 0.0

        # Most confident plate box.
        plate_det = max(dets, key=lambda d: d["confidence"])
        crop = utils.crop_bbox(roi_frame, plate_det["bbox"], pad=0.05)
        if crop is None or crop.size == 0:
            return utils.PLATE_NOT_DETECTED, 0.0

        raw_text = _ocr_text(crop)
        plate = utils.normalize_plate(raw_text)
        if plate == utils.PLATE_NOT_DETECTED:
            return utils.PLATE_NOT_DETECTED, plate_det["confidence"]
        return plate, plate_det["confidence"]
    except Exception:
        return utils.PLATE_NOT_DETECTED, 0.0
