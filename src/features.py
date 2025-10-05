from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer


class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.columns]


def build_baseline_pipeline(df: pd.DataFrame) -> ColumnTransformer:
    # Simple numeric + categorical separate pipelines
    numeric_cols = [c for c in df.columns if df[c].dtype != 'object' and c != 'SalePrice']
    cat_cols = [c for c in df.columns if df[c].dtype == 'object']

    num_pipe = Pipeline([
        ('sel', SelectColumns(numeric_cols)),
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('sel', SelectColumns(cat_cols)),
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer([
        ('num', num_pipe, numeric_cols),
        ('cat', cat_pipe, cat_cols)
    ])


class EngineerFeatures(BaseEstimator, TransformerMixin):
    """Add engineered columns to DataFrame and return augmented frame."""
    def __init__(self):
        self.generated_cols_ = []
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        if 'FullBath' in X.columns and 'HalfBath' in X.columns and 'TotalBath' not in X.columns:
            X['TotalBath'] = X['FullBath'] + 0.5 * X['HalfBath']
        if {'1stFlrSF','2ndFlrSF','TotalBsmtSF'}.issubset(X.columns) and 'TotalSF' not in X.columns:
            X['TotalSF'] = X['1stFlrSF'] + X['2ndFlrSF'] + X['TotalBsmtSF']
        if {'GarageCars','GarageArea'}.issubset(X.columns) and 'GarageQualityRatio' not in X.columns:
            X['GarageQualityRatio'] = X['GarageArea'] / (X['GarageCars'] + 1)
        return X


def build_engineered_pipeline(df: pd.DataFrame) -> Pipeline:
    # Determine column sets AFTER engineering step inside pipeline using a FunctionTransformer-like approach.
    numeric_cols = [c for c in df.columns if df[c].dtype != 'object' and c != 'SalePrice'] + ['TotalBath','TotalSF','GarageQualityRatio']
    # Not all engineered columns may exist; ColumnTransformer will ignore those absent in selection; filter later.
    cat_cols = [c for c in df.columns if df[c].dtype == 'object']

    # Use global DynamicNumericSelector defined below.

    num_pipe = Pipeline([
        ('sel', DynamicNumericSelector(numeric_cols)),
        ('impute', SimpleImputer(strategy='median')),
        ('log', FunctionTransformerSafe(np.log1p)),
        ('scale', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('sel', SelectColumns(cat_cols)),
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    column_transformer = ColumnTransformer([
        ('num', num_pipe, numeric_cols),
        ('cat', cat_pipe, cat_cols)
    ])

    return Pipeline([
        ('eng', EngineerFeatures()),
        ('ct', column_transformer)
    ])

class FunctionTransformerSafe(BaseEstimator, TransformerMixin):
    def __init__(self, func):
        self.func = func
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        import numpy as np
        # X is numpy array after imputation; apply log1p safely element-wise where >0
        X = np.array(X, copy=True)
        positive_mask = X > 0
        X[positive_mask] = self.func(X[positive_mask])
        return X


class DynamicNumericSelector(SelectColumns):
    def transform(self, X):
        cols = [c for c in self.columns if c in X.columns]
        # Filter numeric only
        import pandas as pd
        df = X[cols]
        keep = []
        for c in df.columns:
            if not pd.api.types.is_object_dtype(df[c]):
                keep.append(c)
        return df[keep]


def get_feature_pipeline(kind: str, df: pd.DataFrame) -> ColumnTransformer:
    if kind == 'baseline':
        return build_baseline_pipeline(df)
    elif kind == 'engineered':
        return build_engineered_pipeline(df)
    else:
        raise ValueError(f'Unknown feature pipeline kind: {kind}')
