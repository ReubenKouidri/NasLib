import numpy as np
import torch


def get_num_correct(preds, tgts):
    return preds.argmax(dim=1).eq(tgts).sum().item()


class ModelSave:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            if self.verbose:
                self.trace_func("No decrease in val loss...")
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
