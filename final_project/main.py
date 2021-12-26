import torch
import torch.nn as nn
from dataset import ImageDataset
from torch import Tensor
from torch.utils.data.dataloader import DataLoader


class LRModel(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        return nn.Sigmoid()(x)


class Trainer:
    def __init__(
        self,
        in_features: int = 960,
        out_features: int = 20,
        epoch: int = 100,
        alpha: float = 1e-3,
    ) -> None:
        self.epoch = epoch
        self.loss_func = nn.CrossEntropyLoss()
        self.model = LRModel(in_features, out_features)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=alpha)

    def run(self, train_dataloader: DataLoader, test_dataloader: DataLoader):
        for step in range(self.epoch):
            total_loss = 0.0
            idx = 0
            for features, labels in train_dataloader:
                loss = self._train(features, labels)
                total_loss += loss
                if idx % 50 == 0:
                    print(f"[{idx} / {len(train_dataloader)}]   ", f"loss: {loss}")
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


def main(image_dir):
    alpha = 1e-2
    epoch = 1000
    image_dataset = ImageDataset(image_dir)
    train_dataloader, test_dataloader = image_dataset.load_data()
    print(f">>> Successfully load data...")
    trainer = Trainer(epoch=epoch, alpha=alpha, in_features=960, out_features=20)
    print(f">>> Start to train model...")
    trainer.run(train_dataloader, test_dataloader)


if __name__ == "__main__":
    image_dir = "/home/scott/Downloads/faces_4"
    main(image_dir)
