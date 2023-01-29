import torch
import csv
import numpy as np
from my_utils import get_num_correct
from typing import Tuple


def train(model, optimizer, criterion, trainloader, device) -> Tuple[float, float]:
    total_loss = 0
    batches = 0
    total_correct = 0
    model.train()
    for images, targets in trainloader:
        optimizer.zero_grad()
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        preds = model(images)
        total_correct += get_num_correct(preds, targets)
        loss = criterion(preds, targets)
        loss.backward()
        total_loss += loss.item()
        batches += 1
        optimizer.step()

    train_loss = total_loss / batches
    train_acc = total_correct / (batches * trainloader.batch_size)

    return train_loss, train_acc


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


class Trainer:
    def __init__(self):
        self.epoch_id = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_id = 0
        self.run_data = []
        self.run_start_time = None

        self.network = None
        self.loader = None
        self.df = pd.DataFrame()
        self.saver = ModelSave(verbose=True, path=f'checkpoint_run_{self.run_id}.pt')

    def get_results(self):
        print(
            pd.DataFrame.from_dict(
                self.run_data,
                orient='columns'
            ).sort_values("accuracy", axis=0, ascending=False)
            )

    def begin_run(self, run, network, loader, lead, wavelet) -> None:
        self.run_start_time = time.time()
        self.run_params = run
        self.run_id += 1
        self.network = network
        self.loader = loader
        self.lead = lead
        self.wavelet = wavelet

    def end_run(self) -> None:
        self.epoch_id = 0

    def begin_epoch(self) -> None:
        self.epoch_start_time = time.time()
        self.epoch_id += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0

    def end_epoch(self) -> None:
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / 100  # len(self.loader)
        accuracy = self.epoch_num_correct / 100  # len(self.loader)

        results = OrderedDict()
        results["run"] = self.run_id
        results["epoch"] = self.epoch_id
        results["loss"] = loss
        results["accuracy"] = accuracy
        results["epoch duration"] = epoch_duration
        results["run duration"] = run_duration
        self.saver.path = f'/content/gdrive/MyDrive/new_checkpoint_run_{self.run_id}_wavelet_{self.wavelet}_lead{self.lead}.pt'
        self.saver(loss, self.network)
        for k, v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)
        self.df = self.df.from_dict(
            self.run_data, orient='columns'
            )  # pd.DataFrame.from_dict(self.run_data, orient='columns')
        print(self.df.head())

    def track_loss(self, loss, batch) -> None:
        self.epoch_loss += loss.item() * batch[0].shape[0]

    def track_num_correct(self, preds, labels) -> None:
        self.epoch_num_correct += self._get_num_correct(preds, labels)

    @torch.no_grad()
    def _get_num_correct(self, preds, labels) -> int:
        return preds.argmax(dim=1).eq(labels).sum().item()

    def save(self, fileName) -> None:
        pd.DataFrame.from_dict(
            self.run_data,
            orient='columns',
        ).to_csv(f'{fileName}.csv')
