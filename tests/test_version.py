import unittest

import pywavelet


class VersionTestCase(unittest.TestCase):
    """ Version tests """

    def test_version(self):
        """ check pywavelet exposes a version attribute """
        self.assertTrue(hasattr(pywavelet, "__version__"))
        self.assertIsInstance(pywavelet.__version__, str)


if __name__ == "__main__":
    unittest.main()
