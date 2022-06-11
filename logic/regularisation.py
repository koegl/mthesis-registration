# https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
import torch


class LRScheduler:
    """
    If the validation loss does not decrease for the given number of 'patience' epochs, the learning rate is reduced by
    a factor gamma.
    """

    def __init__(self, optimizer, patience=1, min_lr=1e-4, factor=0.1):
        """
        new_lr = lr * gamma
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor

        self.lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=self.patience,
            min_lr=self.min_lr,
            factor=self.factor,
            verbose=True
        )

    def __call__(self, val_loss, epoch):
        self.lr_schedule.step(val_loss, epoch)


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience and delta"
    """

    def __init__(self, patience=5, min_delta=0.02):
        """
        :param patience: number of epochs to wait before early stopping
        :param min_delta: minimum difference between new loss and old loss for new loss to be considered as an
        improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss

        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0

        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True
