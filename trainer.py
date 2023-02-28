import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from my_utils.ksplit import load_2d_dataset, split_dataset
from my_utils.my_utils import get_num_correct
from my_utils.config import Config
import torch.nn as nn
from tqdm import tqdm
import csv
from typing import overload
from torchvision.utils import make_grid


class Trainer:
    def __init__(self,
                 config: Config,
                 checkpoint_dir: str = 'checkpoints',
                 split_ratio: tuple[int, tuple] | None = None
                 ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.data = self._load_data(self.config.data.data_path, self.config.data.reference_path, split_ratio)
        self.split_ratio = split_ratio
        #self.writer = SummaryWriter(log_dir='logs')

    def _train(self, model, trainloader, optimizer, criterion) -> tuple[float, float]:
        model.train()
        total_loss = 0.0
        correct = 0
        trainloader.dataset.dataset.test = False
        for imgs, tgts in tqdm(trainloader):
            optimizer.zero_grad()
            imgs = imgs.to(self.device, non_blocking=True)
            tgts = tgts.to(self.device, non_blocking=True)
            preds = model(imgs)
            loss = criterion(preds, tgts)
            loss.backward()
            correct += get_num_correct(preds, tgts)
            total_loss += loss.item()
            optimizer.step()

        train_loss = total_loss / (len(trainloader.dataset) / trainloader.batch_size)
        train_acc = correct / (len(trainloader.dataset))

        return train_loss, train_acc

    @torch.inference_mode()
    def _eval(self, model, valloader, criterion) -> tuple[float, float]:
        model.eval()
        total_loss = 0.0
        correct = 0
        valloader.dataset.dataset.test = False
        for imgs, tgts in tqdm(valloader):
            imgs = imgs.to(self.device, non_blocking=True)
            tgts = tgts.to(self.device, non_blocking=True)
            preds = model(imgs)
            correct += get_num_correct(preds, tgts)
            loss = criterion(preds, tgts)
            total_loss += loss.item()

        val_acc = correct / len(valloader.dataset)
        val_loss = total_loss / (len(valloader.dataset) / valloader.batch_size)
        return val_loss, val_acc

    @torch.inference_mode()
    def test(self, model, testloader, answer_path):
        model.eval()
        testloader.dataset.dataset.test = True
        score_matrix = torch.zeros((9, 9), dtype=torch.float32, device=self.device)
        for imgs, tgts in testloader:
            imgs, tgts = imgs.to(self.device), tgts.to(self.device)
            preds = model(imgs).argmax(dim=1)
            print(preds)
            #assert all(0 < pred < 9 and not torch.isnan(pred) for pred in preds)
            for pred, tgt in zip(preds, tgts):
                if pred in tgt:
                    score_matrix[pred, pred] += 1
                else:
                    score_matrix[tgt[0], pred] += 1

        F1 = torch.mean(2 * score_matrix.diag() / (score_matrix.sum(dim=1) + score_matrix.sum(dim=0)))

        Faf = 2 * score_matrix[1, 1].item() / (
                    torch.sum(score_matrix[1, :]).item() + torch.sum(score_matrix[:, 1]).item())
        Fblock = 2 * torch.sum(score_matrix[2:5, 2:5]).item() / (
                    torch.sum(score_matrix[2:5, :]).item() + torch.sum(score_matrix[:, 2:5]).item())
        Fpc = 2 * torch.sum(score_matrix[5:7, 5:7]).item() / (
                    torch.sum(score_matrix[5:7, :]).item() + torch.sum(score_matrix[:, 5:7]).item())
        Fst = 2 * torch.sum(score_matrix[7:9, 7:9]).item() / (
                    torch.sum(score_matrix[7:9, :]).item() + torch.sum(score_matrix[:, 7:9]).item())

        with open(f"{answer_path}.txt", 'w') as score_file:
            print(f"Total File Number: {torch.sum(score_matrix).item()}", file=score_file)
            print(f"F1: {F1:.3f}", file=score_file)
            print(f"Faf: {Faf:.3f}", file=score_file)
            print(f"Fblock: {Fblock:.3f}", file=score_file)
            print(f"Fpc: {Fpc:.3f}", file=score_file)
            print(f"Fst: {Fst:.3f}", file=score_file)

    @staticmethod
    def _load_data(dp, rp, ksplit):
        dataset = load_2d_dataset(dp, rp)
        if ksplit:
            datasets = split_dataset(dataset, ksplit)
            return datasets
        return dataset

    def save_model(self): ...

    @overload
    def track_metrics(self, tr_loss, tr_acc, rep, epoch): ...

    @overload
    def track_metrics(self, tr_loss, tr_acc, ev_loss, ev_acc, rep, epoch): ...

    def track_metrics(self, *args):
        if len(args) == 6:
            self.writer.add_scalar(f"trainingLoss/{args[-2]}", args[0], args[-1])
            self.writer.add_scalar(f"trainingAcc/{args[-2]}", args[1], args[-1])
            self.writer.add_scalar(f"validationLoss/{args[-2]}", args[2], args[-1])
            self.writer.add_scalar(f"validationAcc/{args[-2]}", args[3], args[-1])
        elif len(args) == 4:
            self.writer.add_scalar(f"trainingLoss/{args[-2]}", args[0], args[-1])
            self.writer.add_scalar(f"trainingAcc/{args[-2]}", args[1], args[-1])

    def __call__(self, model, epochs, output):
        for fold, triplet in enumerate(self.data):
            print(f"Starting training on fold number {fold + 1} / {self.split_ratio[0]}\n")
            assert len(triplet) == 3
            assert isinstance(triplet, tuple)

            """
            valset = triplet[1]  # validation set
            img_tgt_pair = triplet[1][0]
            indices = valset.indices
            imgs = [valset.dataset[index][0] for index in indices]
            self.writer.add_graph(model.eval(), img_tgt_pair[0].unsqueeze(dim=0))
            self.writer.add_image(tag="images", img_tensor=make_grid(imgs))
            self.writer.close()
            import sys
            sys.exit()
            """

            train_loader = DataLoader(dataset=triplet[0], batch_size=self.config.train.batch_size, shuffle=True) if \
                self.split_ratio[1][0] > 0 else None
            eval_loader = DataLoader(dataset=triplet[1], batch_size=len(triplet[1]), shuffle=False) if \
                self.split_ratio[1][1] > 0 else None
            test_loader = DataLoader(dataset=triplet[2], batch_size=len(triplet[2]), shuffle=False) if \
                self.split_ratio[1][2] > 0 else None

            optimizer = SGD(model.parameters(), self.config.train.lr, momentum=self.config.train.momentum,
                            nesterov=self.config.train.nesterov)

            criterion = nn.CrossEntropyLoss()

            rep = f"fold{fold}_lr_{self.config.train.lr}"
            for epoch in range(epochs):

                print(f"\nEpoch {epoch + 1}:\n Training...")
                tr_loss, tr_acc = self._train(model, train_loader, optimizer, criterion)

                print("\nvalidating...\n")
                if eval_loader:
                    v_loss, v_acc = self._eval(model, eval_loader, criterion)
                    #self.track_metrics(tr_loss, tr_acc, v_loss, v_acc, rep, epoch)
                else:
                    pass
                    #self.track_metrics(tr_loss, tr_acc, rep, epoch)

            if test_loader:
                self.test(model, test_loader, f"answers_{rep}.csv")
