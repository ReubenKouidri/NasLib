from models import model
import torch
from my_datasets.CPSCDataset import CPSCDataset2D
from torch.utils.data import DataLoader
from torch.optim import SGD
import torch.nn.functional as F
from typing import Any


TRAIN_PATH = "/Users/juliettekouidri/Documents/Reuben/Projects/Python/NasLib/my_datasets/cpsc_data/Train300"
REF_PATH = "/Users/juliettekouidri/Documents/Reuben/Projects/Python/NasLib/my_datasets/cpsc_data/reference300.csv"
EPOCHS = 25
BATCH_SIZE = 20
SHUFFLE = True
NESTEROV = True
MOMENTUM = 0.9
LR = 0.01



def kfold_split(dataset, k, ratio = 0.80):
    splits = []
    for _ in range(k):
        train, eval, test = torch.utils.data.random_split(dataset, (ratio, round((1-ratio)/2, 1), round((1-ratio)/2, 1)))
        splits.append([train, eval, test])

    return splits


trainset = CPSCDataset2D(TRAIN_PATH, REF_PATH)
splits = kfold_split(trainset, 10, 0.8)
for split in splits:
    print(len(split[0]), " ", len(split[1]), " ", len(split[2]))





"""model = model.Model()
trainset = CPSCDataset2D(TRAIN_PATH, REF_PATH)
trainloader = DataLoader(trainset, BATCH_SIZE, SHUFFLE)
optimizer = SGD(model.parameters(), LR, momentum=MOMENTUM, nesterov=NESTEROV)


for epoch in range(EPOCHS):
    epoch_loss = 0
    num_correct = 0
    total = 0

    for i, batch in enumerate(trainloader):
        print("batch ", i + 1, " epoch ", epoch)
        imgs, tgts = batch[0], batch[1]
        preds = model(imgs)
        loss = F.cross_entropy(preds, tgts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        total += imgs.size(0)
        num_correct += preds.argmax(dim=1).eq(tgts).sum().item()

    print("end of epoch ", epoch, ". Loss: ", epoch_loss, ". Num correct: ", num_correct, "/", total)
"""