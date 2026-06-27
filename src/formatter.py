# -*- coding: utf-8 -*-
"""
Formatter: turns aggregated raw results into the EXACT competition JSON schema.

Guarantees:
  - JSON keys match the document verbatim (confidence_score, zaman_saniye, ...).
  - Only whitelisted labels survive; invalid ones are dropped.
  - All text values are ASCII-safe and lower-case; plaka is upper-case.
  - confidence_score is a float clipped to [0.0, 1.0].
  - zaman_saniye is a float.
  - arac_bilgisi contains ONLY the four schema keys: tip, plaka, renk, confidence_score.
  - Repeated detections of the same label are temporally filtered.
"""

try:
    from src import utils
except ModuleNotFoundError:
    import utils

# Whitelists — verbatim from the competition document.
VEHICLE_TYPES = {"sedan", "suv", "hatchback", "pickup", "minibus", "panelvan", "kamyon"}
COLORS = {"beyaz", "siyah", "gri", "kirmizi", "mavi", "sari", "yesil", "turuncu", "kahverengi"}
DRIVER_ACTIONS = {
    "arkaya_bakma", "esneme", "sigara_icme", "su_icme",
    "telefonla_konusma", "slalom", "etrafa_bakinma", "emniyet_kemeri_ihlali",
}
OBJECTS = {"teknocan", "bilgisayar"}
PASSENGERS = {"arka_koltuk_1", "arka_koltuk_2", "on_koltuk"}

CATEGORY_WHITELIST = {
    "sofor_eylemi": DRIVER_ACTIONS,
    "nesneler": OBJECTS,
    "yolcular": PASSENGERS,
}

# Fallbacks when nothing valid was detected — keep schema valid.
DEFAULT_TYPE = "sedan"
DEFAULT_COLOR = "gri"

TEMPORAL_WINDOW_SECONDS = 1.5


def format_output(raw_predictions, video_id=""):
    """
    Convert raw pipeline predictions into the competition-compliant JSON structure.

    raw_predictions = {
        "tip": str | None,
        "renk": str | None,
        "plaka": str,
        "arac_confidence": float,
        "tespitler": [
            {"zaman_saniye": float, "kategori": str, "etiket": str, "confidence_score": float},
            ...
        ]
    }

    Returns:
    {
        "video_id": str,
        "arac_bilgisi": {"tip", "plaka", "renk", "confidence_score"},
        "tespitler": [...]
    }
    """
    if not isinstance(raw_predictions, dict):
        raw_predictions = {}

    # ── arac_bilgisi ────────────────────────────────────────────────────────────
    tip = utils.to_ascii(raw_predictions.get("tip"))
    if tip not in VEHICLE_TYPES:
        tip = DEFAULT_TYPE

    renk = utils.to_ascii(raw_predictions.get("renk"))
    if renk not in COLORS:
        renk = DEFAULT_COLOR

    raw_plaka = raw_predictions.get("plaka") or utils.PLATE_NOT_DETECTED
    plaka = str(raw_plaka).upper().strip()

    # Exactly four keys — no extras.
    arac_bilgisi = {
        "tip": tip,
        "plaka": plaka,
        "renk": renk,
        "confidence_score": _clamp_round(raw_predictions.get("arac_confidence", 0.0)),
    }

    # ── tespitler ───────────────────────────────────────────────────────────────
    cleaned = []
    for d in raw_predictions.get("tespitler", []):
        try:
            kategori = utils.to_ascii(d.get("kategori"))
            valid_labels = CATEGORY_WHITELIST.get(kategori)
            if valid_labels is None:
                continue

            etiket = utils.to_ascii(d.get("etiket"))
            if etiket not in valid_labels:
                continue

            zaman = round(float(d["zaman_saniye"]), 1)

            # Exactly four keys — no hiz_kmh or any other extra.
            cleaned.append({
                "zaman_saniye": zaman,
                "kategori": kategori,
                "etiket": etiket,
                "confidence_score": _clamp_round(d.get("confidence_score", 0.0)),
            })
        except (TypeError, ValueError, KeyError):
            continue

    tespitler = utils.temporal_filter(cleaned, window_seconds=TEMPORAL_WINDOW_SECONDS)

    return {
        "video_id": video_id,
        "arac_bilgisi": arac_bilgisi,
        "tespitler": tespitler,
    }


def _clamp_round(value, ndigits=2):
    return round(utils.clamp_confidence(value), ndigits)


# Keep old name as alias so any external callers don't break.
format_results = format_output
