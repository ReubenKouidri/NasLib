import pytest

from dnasty.utils.config import Config, _convert_type


def test_basic_config():
    config = Config({"key1": "value1", "key2": "2", "key3": "3.0", "key4": 4})
    assert config.key1 == "value1"
    assert config.key2 == 2
    assert config.key3 == 3.0
    assert config.key4 == 4


def test_nested_config():
    config = Config(
        {
            "section1": {"key1": "true", "key2": "false"},
            "section2": {"key3": "None", "key4": "text"},
        }
    )
    assert config.section1.key1 is True
    assert config.section1.key2 is False
    assert config.section2.key3 is None
    assert config.section2.key4 == "text"


def test_list_handling():
    config = Config({"list_section": ["1", "2.0", "true", "none", "text"]})
    assert config.list_section == [1, 2.0, True, None, "text"]


def test_convert_type():
    assert _convert_type("10") == 10
    assert _convert_type("3.14") == 3.14
    assert _convert_type("true") is True
    assert _convert_type("false") is False
    assert _convert_type("None") is None
    assert _convert_type("text") == "text"
    assert _convert_type(7) == 7


def test_missing_attribute():
    config = Config({"key": "value"})
    with pytest.raises(AttributeError):
        _ = config.missing_key
    assert config.get("missing_key", "d") == "d"
    assert "key" in config


def test_incorrect_initialization():
    with pytest.raises(TypeError):
        Config(123)


def test_yaml_and_to_dict(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("a:\n  b: 1\n  c: [x, '2']\nseed: 0\n")
    cfg = Config.from_file(path)
    assert cfg.a.b == 1
    assert cfg.a.c == ["x", 2]
    assert cfg.to_dict() == {"a": {"b": 1, "c": ["x", 2]}, "seed": 0}
