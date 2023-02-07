import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    def __init__(self, model, dataloaders, loss_fn, optimizer, metric_fn, device, checkpoint_dir):
        self.model = model
        self.dataloaders = dataloaders
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric_fn = metric_fn
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.training_loss = []
        self.validation_loss = []
        self.training_metrics = []
        self.validation_metrics = []

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            epoch_training_loss = 0
            epoch_training_metrics = 0
            self.model.train()
            for inputs, labels in self.dataloaders['train']:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_training_loss += loss.item()
                epoch_training_metrics += self.metric_fn(outputs, labels)

            self.training_loss.append(epoch_training_loss / len(self.dataloaders['train']))
            self.training_metrics.append(epoch_training_metrics / len(self.dataloaders['train']))

            epoch_validation_loss = 0
            epoch_validation_metrics = 0
            self.model.eval()
            with torch.no_grad():
                for inputs, labels in self.dataloaders['val']:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels)
                    epoch_validation_loss += loss.item()
                    epoch_validation_metrics += self.metric_fn(outputs, labels)

            self.validation_loss.append(epoch_validation_loss / len(self.dataloaders['val']))
            self.validation_metrics.append(epoch_validation_metrics / len(self.dataloaders['val']))

            if (epoch+1) % 10 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'training_loss': self.training_loss,
                    'validation_loss': self.validation_loss,
                    'training_metrics': self.training_metrics,
                    'validation_metrics': self.validation_metrics,
                }, self.checkpoint_dir + '/checkpoint_' + str(epoch + 1) + '.pth')

    def test(self):
        self.model.eval()
        with torch.no_grad():
            test_loss = 0
            test_metrics = 0
            for inputs, labels in self.dataloaders['test']:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                test_loss += loss.item()
                test_metrics += self.metric_fn(outputs, labels)

        test_loss = test_loss / len(self.dataloaders['test'])
        test_metrics = test_metrics / len(self.dataloaders['test'])
        print('Test Loss: {:.4f}'.format(test_loss))
        print('Test Metrics: {:.4f}'.format(test_metrics))
