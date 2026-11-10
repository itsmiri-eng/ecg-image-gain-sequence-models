"""The v1 cache must open after a move to another computer, and only the machine-specific fields may differ."""
import copy
import unittest
from pathlib import Path
from unittest import mock

from src import portable_cache
from src import ptbxl_learning_data as v1

ELSEWHERE = "C:/Users/someone/Documents/ecg swiss/data/derived/ptbxl_images_v1"


@unittest.skipUnless((v1.DEFAULT_CACHE / "protocol.json").exists(), "v1 learning cache not present")
class TestPortableCache(unittest.TestCase):
    def moved(self, **runtime):
        """Validate as if the cache had been built at ELSEWHERE with a different runtime."""
        real = v1._read_json

        def fake(path):
            value = copy.deepcopy(real(path))
            if Path(path).parent == v1.DEFAULT_CACHE and path.name in ("protocol.json", "metadata.json"):
                value["binding"]["source_root"] = ELSEWHERE
                value["binding"]["runtime"].update(runtime)
            return value
        with mock.patch.object(v1, "_read_json", side_effect=fake), \
                mock.patch.object(portable_cache, "relocate", return_value=v1.DEFAULT_SOURCE):
            return portable_cache.validate_cache(v1.DEFAULT_CACHE)

    def test_relocation_maps_into_this_project(self):
        self.assertEqual(portable_cache.relocate("/Users/x/old/ecg swiss/data/derived/ptbxl_images_v1"), v1.DEFAULT_SOURCE)

    def test_moved_cache_opens_with_same_rows(self):
        rows, _, check = self.moved(python="3.12.9 (main) [MSC v.1942 64 bit (AMD64)]")
        self.assertEqual(check["mode"], "relocated")
        built_at = Path(v1._read_json(v1.DEFAULT_CACHE / "protocol.json")["binding"]["source_root"])
        if not built_at.exists():
            self.skipTest("the strict v1 loader only runs on the computer that built the cache")
        strict_rows, _ = v1._validate_cache(v1.DEFAULT_CACHE)
        self.assertEqual(rows, strict_rows)

    def test_other_runtime_differences_still_refused(self):
        for runtime in ({"numpy": "1.26.4"}, {"opencv": "4.10.0"}):
            with self.subTest(runtime=runtime), self.assertRaisesRegex(ValueError, "binding changed"):
                self.moved(**runtime)


if __name__ == "__main__":
    unittest.main()
