import os
import pickle

from typing import List

from feature_engine.selection import DropFeatures

import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class Solution:
    """
    Class for solver of the Heart Disease UCI problem
    """
    def __init__(
        self,
        exclude: List,
        model: str,
        preprocessing: List,
        kernel: str,
        C: float,
        penalty: float,
        n_estimators: int,
        max_depth: int,
    ):
        self.model = model
        preprocessing = getattr(sklearn.preprocessing, preprocessing) 

        if self.model == "SVC":
            self.C = C
            self.kernel = kernel
            self.pipeline = make_pipeline(
                DropFeatures(exclude), preprocessing(), SVC(kernel=kernel, C=C)
            )
        if self.model == "RandomForest":
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.pipeline = make_pipeline(
                DropFeatures(exclude),
                preprocessing(),
                RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth),
            )
        if self.model == "LogisticRegression":
            self.C = C
            self.penalty = penalty
            self.pipeline = make_pipeline(
                DropFeatures(exclude),
                preprocessing(),
                LogisticRegression(C=C, penalty=penalty),
            )

    def fit(self, x_train, y_train):
        self.pipeline.fit(x_train, y_train)

    def predict(self, x_test):
        return self.pipeline.predict(x_test)

    def save(self, path_to_save: str, name: str = None):
        if name is not None:
            with open(os.path.join(path_to_save, name), "wb") as file:
                pickle.dump(self.pipeline, file)
        else:
            if self.model == "SVC":
                with open(
                    os.path.join(
                        path_to_save, f"{self.model}_{self.C}_{self.kernel}.pkl"
                    ),
                    "wb",
                ) as file:
                    pickle.dump(self.pipeline, file)
            elif self.model == "RandomForest":
                with open(
                    os.path.join(
                        path_to_save,
                        f"{self.model}_{self.n_estimators}_{self.max_depth}.pkl",
                    ),
                    "wb",
                ) as file:
                    pickle.dump(self.pipeline, file)
            elif self.model == "LogisticRegression":
                with open(
                    os.path.join(
                        path_to_save, f"{self.model}_{self.C}_{self.penalty}.pkl"
                    ),
                    "wb",
                ) as file:
                    pickle.dump(self.pipeline, file)

def load_model(path_to_model: str):
    with open(path_to_model, "rb") as file:
        model = pickle.load(file)
    return model

def write_preds(preds, path_to_save: str, name: str = None):
    if name is not None:
        with open(os.path.join(path_to_save, name), "w") as file:
            for item in preds:
                file.write("%s\n" % item)
    else:
        with open(os.path.join(path_to_save, name), "w") as file:
            for item in preds:
                file.write("%s\n" % item)