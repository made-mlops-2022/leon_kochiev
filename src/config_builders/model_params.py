from dataclasses import dataclass, field
from typing import List


@dataclass()
class ModelParams:
    model: str = field(default="SVM")
    C: float = field(default=1.0)
    kernel: str = field(default="rbf")
    penalty: str = field(default="l2")
    random_state_model: int = field(default=228)
    n_estimators: int = field(default=100)
    max_depth: int = field(default=5)


@dataclass()
class PreprParams:
    preprocessing: str = field(default="StandardScaler")
    random_state_prep: int = field(default=228)
    