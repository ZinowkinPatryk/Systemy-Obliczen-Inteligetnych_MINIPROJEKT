import timm
import torch
import torch.nn as nn
from dataset_manager import DatasetManager
from torchvision import transforms


class ModelManager:
    def __init__(self, model_name="resnet18"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history ={"train_lost": [], "val_loss": [], "train_acc": [], "val_acc": [], "test_loss": [], "test_acc": []}
        self.mode = timm.create_model(model_name, pretrained=True, num_classes=10)
        self.dataset_manager = DatasetManager()

    def set_trainable_layers(self, stage:str):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def step_forward(self):
        pass

    def test_results(self):
        pass

    def configure_optimizers(self):
        pass

    def get_prediction(self):
        pass

    def save_history(self):
        pass







