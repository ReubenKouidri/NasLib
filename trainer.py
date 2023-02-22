import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from my_utils.ksplit import load_2d_dataset, split_dataset
from my_utils.my_utils import get_num_correct
from my_utils.config import Config
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import make_grid


class Trainer:
    def __init__(self,
                 config: Config,
                 checkpoint_dir: str = 'checkpoints',
                 ksplit: tuple[int, tuple] | None = None
                 ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.data = self._load_data(self.config.data.data_path, self.config.data.reference_path, ksplit)
        self.ksplit = ksplit
        self.writer = SummaryWriter(log_dir='logs')

    def _train(self, model, trainloader, optimizer, criterion) -> tuple[float, float]:
        model.train()
        total_loss = 0.0
        correct = 0
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
    def test(self, model):
        ...

    @staticmethod
    def _load_data(dp, rp, ksplit):
        dataset = load_2d_dataset(dp, rp)
        if ksplit:
            datasets = split_dataset(dataset, ksplit[0], ksplit[1])
            return datasets
        return dataset

    def save_model(self): ...

    def __call__(self, model):
        for fold, triplet in enumerate(self.data):
            print(f"Starting training on fold number {fold + 1} / {self.ksplit[0]}\n")
            assert len(triplet) == 3
            assert isinstance(triplet, tuple)

            """valset = triplet[1]  # validation set
            img_tgt_pair = triplet[1][0]
            indices = valset.indices
            imgs = [valset.dataset[index][0] for index in indices]
            self.writer.add_graph(model.eval(), img_tgt_pair[0].unsqueeze(dim=0))
            self.writer.add_image(tag="images", img_tensor=make_grid(imgs))
            self.writer.close()
            import sys
            sys.exit()"""

            train_loader = DataLoader(dataset=triplet[0], batch_size=self.config.train.batch_size, shuffle=True)
            eval_loader = DataLoader(dataset=triplet[1], batch_size=len(triplet[1]), shuffle=False)
            test_loader = DataLoader(dataset=triplet[2], batch_size=len(triplet[2]), shuffle=False) if \
                len(triplet[2]) > 0 else None
            optimizer = SGD(model.parameters(), self.config.train.lr, momentum=self.config.train.momentum,
                            nesterov=self.config.train.nesterov)

            criterion = nn.CrossEntropyLoss()

            for epoch in range(self.config.train.epochs):
                print(f"\nEpoch {epoch + 1}:\n Training...")
                tr_loss, tr_acc = self._train(model, train_loader, optimizer, criterion)
                print("\nvalidating...\n")
                ev_loss, ev_acc = self._eval(model, eval_loader, criterion)
                self.writer.add_scalar(f'trainingLoss/fold{fold}', tr_loss, epoch)
                self.writer.add_scalar(f'trainingAcc/fold{fold}', tr_acc, epoch)
                self.writer.add_scalar(f'validationLoss/fold{fold}', ev_loss, epoch)
                self.writer.add_scalar(f'validationAcc/fold{fold}', tr_acc, epoch)
