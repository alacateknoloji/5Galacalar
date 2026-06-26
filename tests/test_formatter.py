import unittest

from src import formatter


class FormatterOutputTests(unittest.TestCase):
    def test_preserves_speed_and_passenger_detections(self):
        raw = {
            "tip": "sedan",
            "renk": "beyaz",
            "plaka": "34ABC123",
            "arac_confidence": 0.94,
            "tespitler": [
                {
                    "zaman_saniye": 14.5,
                    "kategori": "sofor_eylemi",
                    "etiket": "slalom",
                    "confidence_score": 0.89,
                    "speed_kmh": 72.4,
                },
                {
                    "zaman_saniye": 22.1,
                    "kategori": "yolcular",
                    "etiket": "on_koltuk",
                    "confidence_score": 0.91,
                },
            ],
        }

        result = formatter.format_results(raw, "video_001.mp4")

        self.assertEqual(result["video_id"], "video_001.mp4")
        self.assertTrue(any(d["etiket"] == "slalom" for d in result["tespitler"]))
        self.assertTrue(any(d["etiket"] == "on_koltuk" for d in result["tespitler"]))
        slalom = next(d for d in result["tespitler"] if d["etiket"] == "slalom")
        self.assertAlmostEqual(slalom["hiz_kmh"], 72.4)


if __name__ == "__main__":
    unittest.main()
