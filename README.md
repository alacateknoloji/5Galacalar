# 5G & AI Akilli Yol Guvenligi - FTR Inference

Video tabanli cikarim sistemi. Tek bir `video.mp4` girdisinden arac bilgisi
(tip, plaka, renk) ve zaman bazli tespitleri (sofor eylemi, nesneler, yolcular)
uretir ve yarisma semasina birebir uygun `results.json` yazar.

## Akis

```
video.mp4
  -> vehicle_type   (arac tespiti; slalom da bu tespitleri kullanir)
  -> slalom         (merkez x salinimi -> sofor_eylemi / slalom)
  -> vehicle_color  (ana arac kirpintisi -> renk)
  -> plate_ocr      (plaka tespiti + OCR + TR regex normalizasyon)
  -> driver_behavior(sofor eylemleri)            [model henuz hazir degil]
  -> object_detection (teknocan, bilgisayar)
  -> passenger_detection (arka_koltuk_1/2, on_koltuk)
  -> formatter      (whitelist + ASCII + clamp + temporal filter)
  -> /app/data/output/results.json
```

## Model dosyalari

`.pt` agirliklarini `models/` icine koyun (Dockerfile bunlari `/app/models/`'e kopyalar):

| Dosya | Durum |
|---|---|
| `vehicle_type.pt` | hazir (arac tespiti; slalom da bunu kullanir) |
| `vehicle_color.pt` | hazir |
| `plate.pt` | hazir |
| `object_detection.pt` | hazir |
| `passenger_detection.pt` | hazir |
| `driver_behavior.pt` | **henuz hazir degil** |

`driver_behavior.pt` gelene kadar o modul guvenli sekilde bos doner; dosyayi
`models/` icine koydugunuz an otomatik aktif olur, baska degisiklik gerekmez.

## Ayarlamaniz gereken yerler (her modulun basinda)

Modellerinizin **kendi sinif adlari** yarisma etiketlerinden farkliysa, ilgili
modulun basindaki `LABEL_MAP` sozlugunu duzenleyin (orn. modeliniz `white`
diyorsa `vehicle_color.py` icinde `"white": "beyaz"`). Agirlik dosya adlari
farkliysa `WEIGHT_FILE` sabitini guncelleyin.

- `src/plate_ocr.py` -> `OCR_BACKEND` ("easyocr" varsayilan, "paddle" alternatif)
- `src/predict.py` -> `ANALYZE_FPS` / `HEAVY_FPS` (10 dk limiti icin kare hizi ayari)
- `src/utils.py` -> `PLATE_NOT_DETECTED` (asagidaki nota bakin)

### Plaka "bulunamadi" degeri hakkinda
Dokuman metni (kural 5.3) bunu **"tespit edilemedi"** (bosluklu) yaziyor; size
verilen prompt ise **"tespit_edilemedi"** (alt cizgi) istiyor. Varsayilan
`tespit_edilemedi`. Hakem hangisini bekliyorsa `utils.py` icindeki tek sabiti
ona gore degistirin.

## Yerelde calistirma (Docker olmadan)

```bash
pip install -r requirements.txt          # + GPU icin uygun torch
# modelleri models/ icine koyun, videoyu data/input/video.mp4 olarak koyun
PYTHONPATH=. python3 - <<'PY'
from src.predict import run_inference
import json
print(json.dumps(run_inference("data/input/video.mp4", "models"), ensure_ascii=False, indent=2))
PY
```

## Docker

```bash
docker build -t teknofest/proje:latest .
docker run --rm --gpus all \
  -v /yol/video.mp4:/app/data/input/video.mp4 \
  -v /yol/cikti:/app/data/output \
  teknofest/proje:latest
```

## Kurallara uyum notu (kural 5.4)

Bu projede hicbir yerde hostname / IP / ortam degiskeni / "degerlendirme
ortaminda miyim" kontrolu yoktur. Model yukleme hatalarinda kod cokmeden devam
eder; bu yalnizca dayaniklilik icindir (dokuman bolum 7 try-except'i oneriyor)
ve davranis her ortamda aynidir.
