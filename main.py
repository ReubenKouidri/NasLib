from models.model import Model
from my_utils.my_utils import get_num_correct, load_datasets
from my_utils.training import train, evaluate
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
import torch.nn as nn
from my_utils.config import Config
import os
import json


local = os.getcwd()
config_path = os.path.join(local, "config.json")


def load_json(json_file: str) -> dict:
    with open(json_file) as fp:
        return json.load(fp)


config = Config(load_json(config_path))

split_accuracies = []

datasets = load_datasets(config.train.data_path, config.train.reference_path)


for i, dataset in enumerate(datasets):
    model = Model()
    train_set, val_set, test_set = dataset
    trainloader = DataLoader(train_set, config.train.batch_size, config.train.shuffle)
    valloader = DataLoader(val_set, batch_size=len(val_set))
    optimizer = SGD(model.parameters(), config.train.lr, momentum=config.train.momentum, nesterov=config.train.nesterov)
    criterion = nn.CrossEntropyLoss()
    device = config.train.device
    epochs = config.train.epochs

    train_losses = []
    train_accuracies = []
    eval_losses = []
    eval_accuracies = []
    for epoch in range(epochs):
        train_loss, train_accuracy = train(model=model, optimizer=optimizer, criterion=criterion, trainloader=trainloader, device=device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        eval_loss, eval_accuracy = evaluate(model, criterion, valloader, device)
        eval_losses.append(eval_loss)
        eval_accuracies.append(eval_accuracy)

    for epoch in range(int(config.train.epochs)):
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
