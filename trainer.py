import torch
from transformations.copy_paste import SimpleCopyPaste, copy_paste
from tqdm import tqdm
import random
import math

class Trainer:

    def __init__(self, model, train_loader, optimizer, lr_scheduler, device, learning_rate):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.copy_paste = SimpleCopyPaste()

    def train_one_epoch(self, epoch, criterion=None, setup_scheduler=False):
        self.model.train()
        epoch_loss = 0  # Initialize the total loss for this epoch
        epoch_loss_model: dict = {}
        progress_bar = tqdm(total=len(self.train_loader), desc="Training")  # Initialize a progress bar
        # Dynamic learning rate
        self.lr_scheduler = None
        if epoch == 1 and setup_scheduler:
            warmup_factor = 1.0 / 1000  # 0.001
            warmup_iters = min(1000, len(self.train_loader) - 1)  # Number of epochs needed to get to 0.001
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=warmup_factor, total_iters=warmup_iters
            )
        for batch_idx, batch in enumerate(self.train_loader):
            # Batch is a pair (images, targets), where:
            # - images is a tensor representing the images contained inside a batch, so has size equal to batch_size
            # - targets is a list of targets, i.e. dictionaries associated to the images inside the batch
            images: torch.Tensor = batch[0]
            targets: list = batch[1]
            # Apply copy and paste from the 201st epoch with probability 25%
            p = random.random()
            if (p > 0.75 and epoch > 200):
                images = torch.unbind(images, dim=0)
                res_images, res_targets = copy_paste(images, targets)
                images = torch.stack(res_images, dim=0)
                targets = res_targets
            images = images.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            loss_dict: dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())  # Sum the losses
            # Update the total loss of the current epoch
            loss_value = losses.item()
            epoch_loss += loss_value
            # Update the total loss for each loss in the model
            for key, value in loss_dict.items():
                if key not in epoch_loss_model:
                    epoch_loss_model[key] = 0
                epoch_loss_model[key] += value.item()
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict)
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            # Summary so update the progress bar
            progress_bar_dict = dict(curr_epoch=epoch, curr_batch=batch_idx, loss=loss_value,
                                     avg_loss=epoch_loss / len(self.train_loader))
            if setup_scheduler:
                progress_bar_dict.update(lr=self.lr_scheduler.get_last_lr()[0])
            progress_bar.set_postfix(progress_bar_dict)
            progress_bar.update()
        # End the training of this epoch
        avg_loss = epoch_loss / len(self.train_loader)
        # Cleanup and close the progress bar
        progress_bar.close()
        return avg_loss, epoch_loss_model, self.lr_scheduler