import torch


def train(model, optimizer, criterion, trainloader):
    model.train()
    total_loss = 0
    epoch_steps = 0

    for batch_idx, (images, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        pred = model(images)

        loss = criterion(pred, labels)
        loss.backward()

        total_loss += loss.item()
        epoch_steps += 1

        optimizer.step()

    return total_loss / epoch_steps


def evaluate(model, criterion, valloader, local_rank):
    # Validation loss
    total_loss = 0.0
    epoch_steps = 0
    total = 0
    correct = 0

    model.eval()
    for i, data in enumerate(valloader, 0):
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.cuda(local_rank), labels.cuda(local_rank)
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


def test_accuracy(model, testset, answer_path, device="cpu"):
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    with torch.no_grad():
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
