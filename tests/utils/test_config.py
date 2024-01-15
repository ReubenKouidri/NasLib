import unittest
from dnasty.my_utils import Config


class TestConfig(unittest.TestCase):
    def test_mapping_input(self):
        config_obj = Config({'key1': 'value1', 'key2': 10})
        self.assertIsInstance(config_obj, Config)
        self.assertEqual(config_obj.key1, 'value1')
        self.assertEqual(config_obj.key2, 10)

    def test_nested_mapping_input(self):
        config_obj = Config({'outer': {'inner': 20.5}})
        self.assertIsInstance(config_obj.outer, Config)
        self.assertEqual(config_obj.outer.inner, 20.5)

    def test_string_conversion(self):
        config_obj = Config({'int_value': 10,
                             'float_value': 20.5,
                             'none_value': None,
                             'str_value': 'text'})
        self.assertEqual(config_obj.int_value, 10)
        self.assertEqual(config_obj.float_value, 20.5)
        self.assertIsNone(config_obj.none_value)
        self.assertEqual(config_obj.str_value, 'text')

    def test_sequence_input(self):
        config_obj = Config([{'key1': 'value1'}, {'key2': 'value2'}])
        self.assertIsInstance(config_obj, list)
        self.assertIsInstance(config_obj[0], Config)
        self.assertEqual(config_obj[0].key1, 'value1')
        self.assertIsInstance(config_obj[1], Config)
        self.assertEqual(config_obj[1].key2, 'value2')

    def test_non_mapping_non_sequence_input(self):
        config_obj = Config('string')
        self.assertEqual(config_obj, 'string')

    def test_invalid_key_access(self):
        config_obj = Config({'key1': 'value1'})
        with self.assertRaises(AttributeError):
            _ = config_obj.nonexistent_key

    def test_from_file(self):
        config_obj = Config.from_file("../tests/utils/test_config.json")
        self.assertIsInstance(config_obj, Config)
        self.assertEqual(config_obj.key1, 1)
        self.assertEqual(config_obj.key2, 2)
        self.assertEqual(config_obj.nested.n1, 1.0)
        self.assertEqual(config_obj.nested.n2, 2.0)


if __name__ == '__main__':
    unittest.main()
