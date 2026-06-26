import os
import tempfile
import unittest

from src import slalom_risk


class SlalomModelPathTests(unittest.TestCase):
    def test_resolve_model_path_uses_shared_vehicle_weight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = os.path.join(tmpdir, "models")
            os.makedirs(models_dir, exist_ok=True)
            weight_path = os.path.join(models_dir, "vehicle_type.pt")
            with open(weight_path, "w", encoding="utf-8"):
                pass

            resolved = slalom_risk.resolve_model_path(models_dir=models_dir)

            self.assertEqual(resolved, weight_path)


if __name__ == "__main__":
    unittest.main()
