from models.model import Model
from my_datasets.CPSCDataset import CPSCDataset2D
from my_utils import kfold_split
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD
import torch.nn.functional as F
from config import Config
import os
import json


def get_num_correct(preds, tgts):
    return preds.argmax(dim=1).eq(tgts).sum().item()


@kfold_split(10)
def get_dataset(datapath, ref_path):
    return CPSCDataset2D(datapath, ref_path)


@torch.no_grad()
def evaluate(model: Model, valloader: DataLoader) -> (torch.float64, float):
    model.eval()
    eval_loss = 0
    eval_correct = 0
    eval_total = 0
    for batch in valloader:
        imgs, tgts = batch[0], batch[1]
        preds = model(imgs)
        eval_loss += F.cross_entropy(preds, tgts).item()
        eval_correct += get_num_correct(preds, tgts)
        eval_total += len(imgs)

    return loss, eval_correct / eval_total


def load(json_file):
    with open(json_file) as fp:
        return json.load(fp)


local = os.getcwd()
JSON = os.path.join(local, "config.json")
#config = Config(load(JSON))

params = load(JSON)
print(sorted(params["Train"].keys()))


"""splits = get_dataset(config.get_value("Train", "file_path"), config.reference_path)
split_accuracies = []

for i, split in enumerate(splits):
    model = Model()
    trainset, valset, testset = split
    trainloader = DataLoader(trainset, config.get_value("Train", "batch_size"), config.shuffle)
    valloader = DataLoader(valset, batch_size=len(valset))
    optimizer = SGD(model.parameters(), config.lr, momentum=config.momentum, nesterov=config.nesterov)

    epoch_accuracies = []

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for i, batch in enumerate(trainloader):
            print("batch ", i + 1, " epoch ", epoch)
            imgs, tgts = batch[0], batch[1]
            preds = model(imgs)
            loss = F.cross_entropy(preds, tgts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += get_num_correct(preds, tgts)
            total += imgs.size(0)

        eval_loss, eval_accuracy = evaluate(model, valloader)

        print("end of epoch ", epoch,
              "Train loss: ", train_loss,
              "\nTrain accuracy : ", train_correct / total,
              "\nEval loss: ", eval_loss,
              "\nEval accuracy: ", eval_accuracy)

        epoch_accuracies.append(eval_accuracy)
    split_accuracies.append(epoch_accuracies)
"""