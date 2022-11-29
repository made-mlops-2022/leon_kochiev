from typing import List


class Validaror:
    def __init__(self):
        pass

    def validate(self, metrics: List, y_true: List, y_pred: List):
        return {metric.__name__: metric(y_true, y_pred) for metric in metrics}
