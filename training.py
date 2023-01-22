import torch
import csv
import numpy as np


def train(model, optimizer, criterion, trainloader, device):
    total_loss = 0
    epoch_steps = 0
    model.train()
    for images, labels in trainloader:
        optimizer.zero_grad()
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds = model(images)
        loss = criterion(preds, labels)
        loss.backward()
        total_loss += loss.item()
        epoch_steps += 1
        optimizer.step()

    return total_loss / epoch_steps


@torch.no_grad()
def evaluate(model, criterion, valloader, device):
    model.eval()

    total_loss = 0.0
    epoch_steps = 0
    total = 0
    correct = 0

    for inputs, labels in valloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        epoch_steps += 1

    val_acc = correct / total
    val_loss = total_loss / epoch_steps
    return val_loss, val_acc


@torch.no_grad()
def test_accuracy(model, testloader, answer_path, device="cpu"):
    with open(answer_path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Recording', 'Result'])
        for data in testloader:
            images, names = data
            images = images.to(device)
            preds = model(images)
            result = torch.argmax(preds, dim=1) + 1

            for i in range(len(names)):
                answer = [names[i], result[i].item()]
                if answer[1] > 9 or answer[1] < 1 or np.isnan(answer[1]):
                    answer[1] = 1
                writer.writerow(answer)
        csvfile.close()
