import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_manager import DatasetManager
from torchinfo import summary
from convert_data_to_torch_format import HFImageDataset, transforms as img_transforms


class ModelManager:
    def __init__(self, model_name="resnet18", stage="head", save_name="test1", batch_size: int = 32):
        self.model_name = model_name
        self.save_model_name = save_name
        self.best_val_loss = float ("inf")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "test_loss": [], "test_acc": []}
        self.model = timm.create_model(model_name, pretrained=True, num_classes=10).to(self.device)
        self.dataset_manager = DatasetManager()
        # ------------------------------------------------------------------------------------
        self.set_trainable_layers(stage)
        self.dataset = self.dataset_manager.split_dataset
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.configure_optimizers()
        self.train_set = DataLoader(
            HFImageDataset(self.dataset["train"], transform=img_transforms),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
        )
        self.evel_set = DataLoader(
            HFImageDataset(self.dataset["validation"], transform=img_transforms),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )
        self.test_set = DataLoader(
            HFImageDataset(self.dataset["test"], transform=img_transforms),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )

    def set_trainable_layers(self, stage: str):
        for param in self.model.parameters():
            param.requires_grad = False
        stages = {
            "head": [self.model.fc],
            "layer4": [self.model.layer4, self.model.fc],
            "layer3_4": [self.model.layer3, self.model.layer4, self.model.fc],
            "all": [self.model],
        }
        if stage not in stages:
            raise Exception("Wrong name")

        for module in stages[stage]:
            for param in module.parameters():
                param.requires_grad = True

    def configure_optimizers(self, lr_backbone: float = 1e-4, lr_head: float = 1e-3, weight_decay: float = 1e-4,):
        head_params = []
        backbone_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("fc"):
                head_params.append(param)
            else:
                backbone_params.append(param)
        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": lr_backbone})
        if head_params:
            param_groups.append({"params": head_params, "lr": lr_head})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        return optimizer

    def step_forward(self, x: torch.Tensor, y: torch.Tensor):
        x = x.to(self.device)
        y = y.to(self.device)
        logits = self.model(x)
        loss = self.criterion(logits, y)
        return loss, logits

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in dataloader:
            self.optimizer.zero_grad(set_to_none=True)
            loss, logits = self.step_forward(x, y)
            loss.backward()
            self.optimizer.step()
            batch_size = x.size(0)
            running_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == y.to(self.device)).sum().item()
            total += batch_size
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def train(self, epochs: int, stage: str = "head", lr: float = 1e-3):
        self.set_trainable_layers(stage)
        self.optimizer = self.configure_optimizers(lr_head=lr)
        for epoch in range(1, epochs + 1):
            loss, acc = self.train_epoch(self.train_set)
            print(f"[{stage}] Epoch {epoch}/{epochs} | Loss: {loss:.4f} | Acc: {acc:.2%}")

    def get_prediction(self):
        pass

    def eval(self):
        pass

    def test_results(self):
        pass

    def save_history(self):
        pass

    def save_model(self):
        torch.save(self.model.state_dict(), f'{self.save_model_name}.pth')

    def __repr__(self):
        return f"Model -> {self.model_name} \n {summary(self.model)}"


if __name__ == "__main__":
    model_manager = ModelManager()
    print(model_manager)
