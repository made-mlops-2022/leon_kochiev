from dataclasses import dataclass, field


@dataclass()
class ValidationParams:
    metrics: str  # = field(default=[f1_score,])
    normalize: bool = field(default=True)
    sample_weight: float = field(default=None)
    target: str = field(default="condition")
    load_split_path: str = field(default="data/inputs/default_splits/test_heart_cleveland_upload.csv")
    load_preds_path: str = field(default="data/outputs/predictions/")
    save_report_path: str = field(default="data/models/reports/")
