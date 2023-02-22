from my_utils.config import Config
from trainer import Trainer
from models.model import Model
import os
import json


local = os.getcwd()
config_path = os.path.join(local, "config.json")


def load_json(json_file: str) -> dict:
    with open(json_file) as fp:
        return json.load(fp)


config = Config(load_json(config_path))
num_splits = 2
ratio_splits = (0.8, 0.2, 0.0)

trainer = Trainer(config=config, ksplit=(num_splits, ratio_splits))

model = Model()

trainer(model)
