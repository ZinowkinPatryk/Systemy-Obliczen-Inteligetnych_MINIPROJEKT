import os
import textwrap
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt
import pandas as pd


class DatasetManager:
    def __init__(self, dataset_size: int = 3000, dataset_path: str ="timm/eurosat-rgb", save_path: str ="../dataset"):
        self.save_path = save_path
        self.dataset_size = dataset_size
        self.data = self._load_dataset(dataset_path)
        self.class_names = self.data["train"].features["label"].names
        self.split_data = None

    @staticmethod
    def _load_dataset(dataset_path: str) -> DatasetDict:
        return load_dataset(dataset_path)

    def plot_samples(self, split="train", save_fig=False):
        labels = list(self.data[split]["label"])
        sample_indices = [labels.index(class_id) for class_id in range(len(self.class_names))]
        fig, axes = plt.subplots(2, 5, figsize=(14, 6))
        fig.suptitle(f"Przykladowe obrazy ze zbioru: {split}", fontsize=14, fontweight="bold")
        for ax, idx in zip(axes.flat, sample_indices):
            sample = self.data[split][idx]
            ax.imshow(sample["image"])
            ax.set_title(self.class_names[sample["label"]], fontsize=11)
            ax.axis("off")
        plt.tight_layout(h_pad=3.0, w_pad=2.0)
        if save_fig:
            os.makedirs(self.save_path, exist_ok=True)
            plt.savefig(os.path.join(self.save_path, "sample_images.png"), dpi=150)
        plt.show()

    def data_visualization(self, save_fig=False):
        self.plot_samples(split="train", save_fig=save_fig)

    def __repr__(self):
        all_labels = [label for split in self.data.values() for label in split["label"]]
        counts = pd.Series(all_labels).value_counts()
        classes_info = "\n".join(f"{name} - {counts.get(idx, 0)}" for idx, name in enumerate(self.class_names))
        data_structure = f"""
        ================================
        Data text information
        ================================
        Dataset structure: 
        {self.data}
        -------------------------------------
        Splits count: 
        {len(self.data)}
        -------------------------------------
        Classes count: 
        {len(self.class_names)}
        -------------------------------------
        Class distribution (all splits):
        {classes_info}
        ================================
        """
        return textwrap.dedent(data_structure).strip()

    def prepare_data(self, seed = 42):
        train_len = int(self.dataset_size * 0.8)
        val_len = int(self.dataset_size * 0.1)
        test_len = self.dataset_size - train_len - val_len
        split_sizes = {
            "train": train_len,
            "validation": val_len,
            "test": test_len
        }
        subsets = {}
        for split_name, size in split_sizes.items():
            dataset = self.data[split_name]
            n_samples = min(size, len(dataset))
            subsets[split_name] = dataset.shuffle(seed=seed).select(range(n_samples))
        self.split_data = DatasetDict(subsets)

    @property
    def split_dataset(self):
        if self.split_data is None:
            self.prepare_data()
        return self.split_data

if __name__ == "__main__":
    dm = DatasetManager(dataset_path="timm/eurosat-rgb")
    print(dm)
    dm.data_visualization()
    print(dm.split_dataset)
