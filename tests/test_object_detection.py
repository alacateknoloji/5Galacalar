import unittest
import numpy as np

from src import object_detection


class ObjectDetectionRoiTests(unittest.TestCase):
    def test_roi_is_cropped_to_vehicle_bbox(self):
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        bbox = [50, 40, 180, 150]

        roi_frame = object_detection._get_detection_frame(frame, bbox)

        self.assertEqual(roi_frame.shape[0], 110)
        self.assertEqual(roi_frame.shape[1], 130)

if __name__ == "__main__":
    unittest.main()
