import unittest

from vision import _snap_to_catalog


class VisionGateTests(unittest.TestCase):
    def test_unknown_block_count_is_not_snapped(self):
        seven = [(0, 0), (0, 1), (0, 2),
                 (1, 0), (1, 1), (1, 2), (2, 0)]
        self.assertIsNone(_snap_to_catalog(seven))

    def test_known_shape_is_normalized(self):
        self.assertEqual(
            _snap_to_catalog([(4, 7), (5, 7), (5, 8)]),
            [(0, 0), (1, 0), (1, 1)],
        )


if __name__ == "__main__":
    unittest.main()
