"""Training pipeline for the tile classifier."""

import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from captcha_solver.config import Config
from captcha_solver.models.utils import save_checkpoint


class Trainer:
    """Trains the tile classifier with validation and checkpointing."""

    def __init__(self, model, train_loader, val_loader, config: Config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )

        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def train(self) -> dict:
        """Run the full training loop. Returns training history."""
        history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
        }
        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(1, self.config.num_epochs + 1):
            train_loss, train_acc = self._train_epoch(epoch)
            val_loss, val_acc = self._validate()
            self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            lr = self.scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch}/{self.config.num_epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} | "
                f"LR: {lr:.6f}"
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    {"val_acc": val_acc, "val_loss": val_loss},
                    os.path.join(self.config.checkpoint_dir, "best_model.pt"),
                )

        history["best_epoch"] = best_epoch
        history["best_val_acc"] = best_val_acc
        print(f"\nBest model at epoch {best_epoch} with val accuracy: {best_val_acc:.2%}")
        return history

    def _train_epoch(self, epoch):
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config.num_epochs} [Train]",
            leave=False,
        )
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)

            pbar.set_postfix(loss=loss.item(), acc=correct / total)

        return total_loss / total, correct / total

    @torch.no_grad()
    def _validate(self):
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)

        return total_loss / total, correct / total
