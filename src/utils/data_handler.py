import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split


class DataHandler:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = pd.read_csv(data_path)

    def split(
        self,
        target_col: str,
        stratify: List[float],
        test_size: float,
        random_state_sampl: int,
    ):
        if stratify == True:
            stratify = self.data[target_col]
        features = list(
            set(self.data.columns)
            - set(
                [
                    target_col,
                ]
            )
        )

        X_train, X_test, y_train, y_test = train_test_split(
            self.data[features],
            self.data[target_col],
            test_size=test_size,
            random_state=random_state_sampl,
            stratify=stratify,
        )

        return X_train, X_test, y_train, y_test
