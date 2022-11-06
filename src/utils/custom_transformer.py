import numpy as np
from numpy.random import rand
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        X_transformed = X_transformed + rand(0, 10, X_transformed.shape)
        return X_transformed