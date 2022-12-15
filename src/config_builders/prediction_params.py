from dataclasses import dataclass, field
from typing import List


@dataclass()
class PredictionParams:
    target: str = field(default="condition")
    load_model_path: str = field(default="data/models/SVC_rbf_1.0_1667761780.9964564.pkl")
    load_split_path: str = field(default="data/inputs/default_splits/test_heart_cleveland_upload.csv")
    save_preds_path: str = field(default="data/outputs/predictions/")
