from my_utils.config import Config
from trainer import Trainer
#from models.modelNew import Model
from models.model import Model
import os
import json


local = os.getcwd()
config_path = os.path.join(local, "config.json")


def load_json(json_file: str) -> dict:
    with open(json_file) as fp:
        return json.load(fp)


config = Config(load_json(config_path))
num_splits = 1
ratio_splits = (0.1, 0.1, 0.8)

trainer = Trainer(config=config, split_ratio=(num_splits, ratio_splits))
model = Model()
trainer(model, epochs=1, output=True)

