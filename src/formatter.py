# -*- coding: utf-8 -*-
"""
Formatter: turns aggregated raw results into the EXACT competition JSON schema.

Guarantees:
  - JSON keys match the document verbatim (confidence_score, zaman_saniye, ...).
  - Only whitelisted labels survive; invalid ones are dropped.
  - All labels are ASCII-safe and lower-case.
  - confidence_score is a float clipped to [0.0, 1.0].
  - zaman_saniye is a float.
  - Repeated detections of the same label are temporally filtered.
"""

try:
    from src import utils
except ModuleNotFoundError:
    import utils

# Whitelists (verbatim from the competition document).
VEHICLE_TYPES = {"sedan", "suv", "hatchback", "pickup", "minibus", "panelvan", "kamyon"}
COLORS = {"beyaz", "siyah", "gri", "kirmizi", "mavi", "sari", "yesil", "turuncu", "kahverengi"}
DRIVER_ACTIONS = {"arkaya_bakma", "esneme", "sigara_icme", "su_icme",
                  "telefonla_konusma", "slalom", "etrafa_bakinma",
                  "bir_sey_icme", "kemer_takili", "mesajlasma"}

OBJECTS = {"teknocan", "bilgisayar"}
PASSENGERS = {"arka_koltuk_1", "arka_koltuk_2", "on_koltuk"}

CATEGORY_WHITELIST = {
    "sofor_eylemi": DRIVER_ACTIONS,
    "nesneler": OBJECTS,
    "yolcular": PASSENGERS,
}

# Fallbacks for the required arac_bilgisi fields when nothing valid was detected.
# These keep the JSON schema-valid; the grader scores them as right/wrong.
DEFAULT_TYPE = "sedan"
DEFAULT_COLOR = "gri"

TEMPORAL_WINDOW_SECONDS = 1.5


def format_results(raw, video_id):
    """
    raw = {
        "tip": str|None, "renk": str|None, "plaka": str,
        "arac_confidence": float,
        "tespitler": [{"zaman_saniye","kategori","etiket","confidence_score"}, ...]
    }
    """
    tip = utils.to_ascii(raw.get("tip"))
    if tip not in VEHICLE_TYPES:
        tip = DEFAULT_TYPE

    renk = utils.to_ascii(raw.get("renk"))
    if renk not in COLORS:
        renk = DEFAULT_COLOR

    plaka = raw.get("plaka") or utils.PLATE_NOT_DETECTED

    arac_bilgisi = {
        "tip": tip,
        "plaka": plaka,
        "renk": renk,
        "confidence_score": clamp_round(raw.get("arac_confidence", 0.0)),
    }

    # Validate / clean every detection.
    cleaned = []
    for d in raw.get("tespitler", []):
        kategori = utils.to_ascii(d.get("kategori"))
        valid_labels = CATEGORY_WHITELIST.get(kategori)
        if valid_labels is None:
            continue
        etiket = utils.to_ascii(d.get("etiket"))
        if etiket not in valid_labels:
            continue
        try:
            zaman = round(float(d.get("zaman_saniye")), 1)
        except (TypeError, ValueError):
            continue
        out_item = {
            "zaman_saniye": zaman,
            "kategori": kategori,
            "etiket": etiket,
            "confidence_score": clamp_round(d.get("confidence_score", 0.0)),
        }
        speed = d.get("speed_kmh")
        if speed is not None:
            try:
                out_item["hiz_kmh"] = round(float(speed), 1)
            except (TypeError, ValueError):
                pass
        cleaned.append(out_item)

    tespitler = utils.temporal_filter(cleaned, window_seconds=TEMPORAL_WINDOW_SECONDS)

    return {
        "video_id": video_id,
        "arac_bilgisi": arac_bilgisi,
        "tespitler": tespitler,
    }


def clamp_round(value, ndigits=2):
    return round(utils.clamp_confidence(value), ndigits)
