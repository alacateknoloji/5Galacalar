import unittest
import numpy as np

from src import driver_behavior


class DriverBehaviorRoiTests(unittest.TestCase):
    def test_default_roi_is_right_side_of_frame(self):
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        x1, y1, x2, y2 = driver_behavior._get_driver_roi(frame)

        self.assertGreater(x1, 0)
        self.assertGreater(x2, x1)
        self.assertGreater(y2, y1)
        self.assertGreater(x1, frame.shape[1] * 0.4)
        self.assertGreater(frame.shape[1], x2)


if __name__ == "__main__":
    unittest.main()
