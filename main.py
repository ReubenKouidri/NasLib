from models.model import Model
from my_datasets.CPSCDataset import CPSCDataset2D
from my_utils import kfold_split, Config
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD
import torch.nn.functional as F
import torch
from typing import Any


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


splits = get_dataset(TRAIN_PATH, REF_PATH)
split_accuracies = []
for i, split in enumerate(splits):
    model = Model()
    trainset, valset, testset = split
    trainloader = DataLoader(trainset, BATCH_SIZE, SHUFFLE)
    valloader = DataLoader(valset, batch_size=len(valset))
    optimizer = SGD(model.parameters(), LR, momentum=MOMENTUM, nesterov=NESTEROV)

    epoch_accuracies = []

    for epoch in range(EPOCHS):
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
