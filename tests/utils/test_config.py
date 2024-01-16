import unittest
from dnasty.my_utils.config import Config, _convert_type


class TestConfig(unittest.TestCase):

    def test_basic_config(self):
        """Test simple configuration loading and attribute access."""
        config_data = {"key1": "value1", "key2": "2", "key3": "3.0"}
        config = Config(config_data)
        self.assertEqual(config.key1, "value1")
        self.assertEqual(config.key2, 2)
        self.assertEqual(config.key3, 3.0)

    def test_nested_config(self):
        """Test nested configuration loading and attribute access."""
        nested_config_data = {
            "section1": {"key1": "true", "key2": "false"},
            "section2": {"key3": "None", "key4": "text"}
        }
        config = Config(nested_config_data)
        self.assertTrue(config.section1.key1)
        self.assertFalse(config.section1.key2)
        self.assertIsNone(config.section2.key3)
        self.assertEqual(config.section2.key4, "text")

    def test_list_handling(self):
        """Test configuration with list values."""
        config_data = {
            "list_section": ["1", "2.0", "true", "none", "text"]
        }
        config = Config(config_data)
        self.assertEqual(config.list_section, [1, 2.0, True, None, "text"])

    def test_convert_type(self):
        """Test the _convert_type helper function."""
        self.assertEqual(_convert_type("10"), 10)
        self.assertEqual(_convert_type("3.14"), 3.14)
        self.assertTrue(_convert_type("true"))
        self.assertFalse(_convert_type("false"))
        self.assertIsNone(_convert_type("None"))
        self.assertEqual(_convert_type("text"), "text")

    def test_missing_attribute(self):
        """Test accessing a non-existent attribute."""
        config_data = {"key": "value"}
        config = Config(config_data)
        with self.assertRaises(AttributeError):
            _ = config.missing_key

    def test_incorrect_initialization(self):
        """Test initialization with incorrect data type."""
        with self.assertRaises(TypeError):
            Config(123)  # Non-mapping type should raise TypeError


if __name__ == '__main__':
    unittest.main()
