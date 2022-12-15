from dataclasses import dataclass, field

@dataclass()
class Paths:
    data_path: str = field(default='data/inputs/heart_cleveland_upload.csv')
    load_model_path: str = field(default='data/models/SVC_1.0_rbf.pkl')
    save_model_path: str = field(default='data/models/')
    save_report_path: str = field(default='data/models/reports/')
    save_preds_path: str = field(default='data/outputs/predictions/')