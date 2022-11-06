from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler 
from typing import List

@dataclass()
class SamplingParams:
    stratify: bool = field(default=True)
    test_size: float = field(default=0.2)
    random_state_sampl : int = field(default=228)
