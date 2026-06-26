Place your trained YOLO weights here. The Dockerfile copies this folder to
/app/models/. Expected filenames (override via WEIGHT_FILE in each module):

  vehicle_type.pt          (vehicle detection; also used by slalom)
  vehicle_color.pt
  plate.pt
  object_detection.pt
  passenger_detection.pt
  driver_behavior.pt       (not ready yet; module stays inactive until present)
