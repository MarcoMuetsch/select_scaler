#IMPORTS

import pandas as pd
from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#LOAD DATA

diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

#TRANSFORMER CLASS

class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler):
        self.scaler = scaler

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.scaler is None:
            X_scaled = X
        elif self.scaler == 'standard':
            scaler = StandardScaler().fit(X)
            X_scaled = scaler.transform(X)
        elif self.scaler == 'minmax':
            scaler = MinMaxScaler().fit(X)
            X_scaled = scaler.transform(X)
        return X_scaled

#PIPELINE

pipe = Pipeline([('scaler', Scaler(None)), ('reg', LinearRegression())])

#GRID SEARCH

grid_params = {'scaler__scaler': [None, 'standard', 'minmax']}

gs = GridSearchCV(pipe, grid_params, scoring='r2')

#SEARCHING

if __name__ == "__main__":

    gs.fit(diabetes_X, diabetes_y)

    df = pd.DataFrame(gs.cv_results_)

    print(df[['param_scaler__scaler', 'mean_test_score']])