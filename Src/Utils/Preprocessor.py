import pandas as pd
import numpy as np
from ..config import cfg
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler


def Remove_outliers(data, column, method='IQR', impute_method='median'):
    return data


def Encode(data, columns, method='label'):
    data = data.copy()
    if method == 'label':
        encoder = LabelEncoder()
        for col in columns:
            data[col] = encoder.fit_transform(data[col])
    elif method == 'onehot':
        data = pd.get_dummies(data, columns=columns, dtype=int)
    else:
        raise Exception('Invalid Type')
    return data


def Scale(data, columns, method='minmax'):
    data = data.copy()
    if method.startswith("log"):
        data[columns] = np.log1p(data[columns])
    if method.endswith("standard"):
        scaler = StandardScaler()
        data[columns] = scaler.fit_transform(data[columns])
    elif method.endswith("minmax"):
        scaler = MinMaxScaler()
        data[columns] = scaler.fit_transform(data[columns])
    elif method.endswith("robust"):
        scaler = RobustScaler()
        data[columns] = scaler.fit_transform(data[columns])
    elif method.endswith("log"):
        return data
    else:
        raise Exception('Invalid Type')
    return data


def Preprocess(data, doOutliers=False, doEncode=False, doScale=False, ScaleMethod='minmax', EncodeMethod='label', columns_encode=cfg.HeartFailure.CATEGORICAL_COLUMNS, columns_scale=cfg.HeartFailure.COLUMNS, impute_method='median', columns_outliers=cfg.HeartFailure.COLUMNS, detect_outlier_method='IQR'):
    if doOutliers:
        for col in columns_outliers:
            data = Remove_outliers(
                data, col, detect_outlier_method, impute_method)
    if doEncode:
        for col in columns_encode:
            data = Encode(data, col, EncodeMethod)

    if doScale:
        for col in columns_scale:
            data = Scale(data, col, ScaleMethod)


combinations = {
    'doOutliers': [True, False],
    'doEncode': [True, False],
    'doScale': [True, False],
    'ScaleMethod': ['minmax', 'standard', 'robust', 'logminmax', 'logstandard', 'logrobust'],
    'EncodeMethod': ['label', 'onehot'],
}
