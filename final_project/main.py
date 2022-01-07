import torch
import torch.nn as nn
from dataset import ImageDataset
from model import CNNModel
from torch import Tensor
from torch.utils.data.dataloader import DataLoader


class Trainer:
    def __init__(self, target_dim: int, epoch: int = 100, alpha: float = 1e-3) -> None:
        self.epoch = epoch
        self.loss_func = nn.CrossEntropyLoss()
        self.model = CNNModel(target_dim)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=alpha)

    def run(self, train_dataloader: DataLoader, test_dataloader: DataLoader):
        for step in range(self.epoch):
            total_loss = 0.0
            idx = 0
            for features, labels in train_dataloader:
                loss = self._train(features, labels)
                total_loss += loss
                idx += 1
            acc = self._test(test_dataloader)
            print(
                f"[{step + 1} / {self.epoch}]   ",
                f"loss: {total_loss / len(train_dataloader)}   ",
                f"acc: {acc}",
            )

    def _train(self, features: Tensor, labels: Tensor) -> float:
        self.optimizer.zero_grad()
        out: Tensor = self.model(features)
        loss: Tensor = self.loss_func(out, labels)
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def _test(self, test_dataloader: DataLoader) -> float:
        total_acc = 0.0
        with torch.no_grad():
            for features, labels in test_dataloader:
                out: Tensor = self.model(features)
                acc = (out.argmax(dim=1) == labels).float().mean()
                total_acc += acc
        return total_acc / len(test_dataloader)


def main(image_dir: str, task: str = "face"):
    assert task in ["face", "emotion"], f"please input valid task name"
    target_dim = {"face": 20, "emotion": 4}
    if task == "emotion":
        alpha = 1e-4
        epoch = 1000
    else:
        alpha = 1e-2
        epoch = 500
    image_dataset = ImageDataset(image_dir)
    train_dataloader, test_dataloader = image_dataset.load_data(target=task)
    print(f">>> Successfully load data...")
    trainer = Trainer(target_dim=target_dim[task], epoch=epoch, alpha=alpha)
    print(f">>> Start to train model...")
    trainer.run(train_dataloader, test_dataloader)


if __name__ == "__main__":
    image_dir = "/home/scott/Downloads/faces_4"
    main(image_dir, task="face")
