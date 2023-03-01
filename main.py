from my_utils.config import Config
from trainer import Trainer
from models.modelNew import Model
import os
import json


local = os.getcwd()
config_path = os.path.join(local, "config.json")


def load_json(json_file: str) -> dict:
    with open(json_file) as fp:
        return json.load(fp)


config = Config(load_json(config_path))
num_splits = 5
ratio_splits = (0.6, 0.2, 0.2)

trainer = Trainer(config=config, split_ratio=(num_splits, ratio_splits))
model = Model()
print(model)
trainer(model, epochs=2, output=True)
