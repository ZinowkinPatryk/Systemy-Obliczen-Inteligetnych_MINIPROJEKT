
class ResultsManager:
    def __init__(self, class_names: list[str], save_dir: str = "results"):
        self.class_names = class_names
        self.save_dir = save_dir

    def plot_learning_curves(self, history: dict, experiment_name: str):
        pass

    def get_classification_metrics(self, y_true, y_pred) -> dict:
        pass

    def plot_confusion_matrix(self, y_true, y_pred, experiment_name: str):
        pass

    def plot_sample_predictions(self, images, y_true, y_pred, correct: bool = False, n: int = 9):
        pass

    def compare_experiments(self, experiments_metrics: dict):
        pass