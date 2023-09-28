import unittest
from dnasty.my_utils.config import Config


class TestConfig(unittest.TestCase):
    def test_mapping_input(self):
        config_obj = Config({'key1': 'value1', 'key2': 'value2'})
        self.assertIsInstance(config_obj, Config)
        self.assertEqual(config_obj.__data, {'key1': 'value1', 'key2': 'value2'})

    def test_mutable_sequence_input(self):
        config_obj = Config([{'key1': 'value1'}, {'key2': 'value2'}])
        self.assertIsInstance(config_obj, list)
        self.assertIsInstance(config_obj[0], Config)
        self.assertEqual(config_obj[0].__data, {'key1': 'value1'})
        self.assertIsInstance(config_obj[1], Config)
        self.assertEqual(config_obj[1].__data, {'key2': 'value2'})

    def test_non_mapping_non_sequence_input(self):
        config_obj = Config('string')
        self.assertEqual(config_obj, 'string')

    def test_keyword_key_in_mapping(self):
        config_obj = Config({'class': 'value'})
        self.assertEqual(config_obj.__data, {'class_': 'value'})

    def test_getattr_method(self):
        config_obj = Config({'key1': {'key2': 'value'}})
        self.assertIsInstance(config_obj.key1, Config)
        self.assertEqual(config_obj.key1.key2, 'value')

    def test_build_method_mapping(self):
        config_obj = Config.build({'key1': 'value1', 'key2': 'value2'})
        self.assertIsInstance(config_obj, Config)
        self.assertEqual(config_obj.__data, {'key1': 'value1', 'key2': 'value2'})

    def test_build_method_mutable_sequence(self):
        config_obj = Config.build([{'key1': 'value1'}, {'key2': 'value2'}])
        self.assertIsInstance(config_obj, list)
        self.assertIsInstance(config_obj[0], Config)
        self.assertEqual(config_obj[0].__data, {'key1': 'value1'})
        self.assertIsinstance(config_obj[1], Config)
        self.assertEqual(config_obj[1].__data, {'key2': 'value2'})

    def test_build_method_non_mapping_non_sequence(self):
        config_obj = Config.build('string')
        self.assertEqual(config_obj, 'string')


if __name__ == '__main__':
    unittest.main()