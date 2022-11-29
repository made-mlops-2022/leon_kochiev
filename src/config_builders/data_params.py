from dataclasses import dataclass, field


@dataclass()
class DataParams:
    exclude: str = field(default="thalach")
    target: str = field(default="condition")
