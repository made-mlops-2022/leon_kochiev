from dataclasses import dataclass, field
from sklearn.metrics import f1_score
from typing import List

@dataclass()
class ValidationParams:
    metrics: str # = field(default=[f1_score,])
    normalize: bool = field(default=True)
    sample_weight: float = field(default=None) #dunno how to hangle NoneType and List simultaneously
