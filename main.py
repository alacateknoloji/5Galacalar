# -*- coding: utf-8 -*-
"""
Entry point.

Reads /app/data/input/video.mp4, runs inference, and writes the consolidated
results to /app/data/output/results.json in the exact competition schema.
Always exits cleanly: even on a fatal error it writes a valid (empty-detection)
results.json so the container never finishes without an output file.
"""

import os
import sys
import json

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from src.predict import run_inference


def _resolve_input_path():
    candidates = [
        os.environ.get("INPUT_PATH"),
        "/app/data/input/video.mp4",                           # judge VM (dokuman Bolum 6)
        os.path.join(ROOT_DIR, "video.mp4"),                   # local dev
        os.path.join(ROOT_DIR, "testvideo5.mp4"),              # local dev (eski isim)
        os.path.join(ROOT_DIR, "data", "input", "video.mp4"),  # local mirrored
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return "/app/data/input/video.mp4"  # fallback to judge path


def _resolve_output_path():
    explicit = os.environ.get("OUTPUT_PATH")
    if explicit:
        return explicit
    return "/app/data/output/results.json"  # dokuman Bolum 6: /app/data/output/results.json


INPUT_PATH = _resolve_input_path()
OUTPUT_PATH = _resolve_output_path()


def _write(data):
    output_dir = os.path.dirname(OUTPUT_PATH) or "."
    os.makedirs(output_dir, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _empty_result():
    """Schema-valid fallback so a results.json always exists."""
    return {
        "video_id": os.path.basename(INPUT_PATH),
        "arac_bilgisi": {
            "tip": "sedan",
            "plaka": "tespit_edilemedi",
            "renk": "gri",
            "confidence_score": 0.0,
        },
        "tespitler": [],
    }


def main():
    print("Road-Safety inference starting...")

    if not os.path.exists(INPUT_PATH):
        print(f"Input video not found -> {INPUT_PATH}")
        _write(_empty_result())
        sys.exit(1)

    try:
        result = run_inference(INPUT_PATH)
        _write(result)
        print(f"Done. Output written: {OUTPUT_PATH}")
    except Exception as e:  # noqa: BLE001 - never crash without an output file
        print(f"Inference failed: {e}")
        _write(_empty_result())
        sys.exit(1)


if __name__ == "__main__":
    main()
