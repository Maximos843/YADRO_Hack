import pandas as pd
import numpy as np
from mlforecast import MLForecast
from window_ops.expanding import expanding_mean
from mlforecast.lag_transforms import RollingMean, ExponentiallyWeightedMean
from mlforecast.target_transforms import LocalStandardScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from config import Models
from tqdm import tqdm
from utils import (create_df_clouds, create_df_humidity, create_df_temps,
                   create_df_wind_speed, get_neibrs, create_df_pressure)


mlf_pressure = MLForecast(
    models=[CatBoostRegressor(**Models.PRESSURE_PARAMS)],
    freq=1,
    lags=[1, 2, 3, 4],
    lag_transforms = {
        1:  [expanding_mean],
        2: [expanding_mean],
    },
)

mlf_forecast_clouds = MLForecast(
    models=[CatBoostRegressor(**Models.CLOUDS_PARAMS)],
    freq=1,
    lags=list(range(1, 5)),
    lag_transforms = {
        1:  [expanding_mean],
        2: [expanding_mean],
        3: [ExponentiallyWeightedMean(alpha=0.2)],
    },
    target_transforms=[LocalStandardScaler()]
)


mlf_forecast_speed = MLForecast(
    models=[XGBRegressor(**Models.WIND_SPEED_PARAMS)],
    freq=1,
    lags=list(range(1, 4)),
    lag_transforms = {
        1: [ExponentiallyWeightedMean(alpha=0.25)],
        2: [ExponentiallyWeightedMean(alpha=0.2)],
        3: [ExponentiallyWeightedMean(alpha=0.2)],
        4: [RollingMean(window_size=5)],
        5: [RollingMean(window_size=5)],
    },
    target_transforms=[LocalStandardScaler()]
)

mlf_forecast_humidity = MLForecast(
    models=[CatBoostRegressor(**Models.HUMIDITY_PARAMS)],
    freq=1,
    lags=[1, 2, 3, 4],
    lag_transforms = {
        1: [expanding_mean, RollingMean(window_size=5)],
        2: [expanding_mean],
        4: [RollingMean(window_size=5)],
    },
    target_transforms=[LocalStandardScaler()]
)

mlf_forecast_temps = MLForecast(
    models=[CatBoostRegressor(**Models.TEMPRETURE_PARAMS)],
    freq=1,
    lags=[1, 2, 3, 4],
    lag_transforms = {
        1:  [expanding_mean, ExponentiallyWeightedMean(alpha=0.3)],
        2: [expanding_mean, RollingMean(window_size=5)],
    },
)


def make_all_forecast_speed(wind_speed: np.array) -> np.array:
    for hour in tqdm(range(1, 6)):
        preds = make_current_hour_preds(wind_speed, hour, 'speed')
        wind_speed = wind_speed.tolist()
        wind_speed.append([[preds[30 * row + col] for col in range(30)] for row in range(30)])
        wind_speed = np.array(wind_speed)
    return np.array([wind_speed[i] for i in range(43, 48)]).reshape(-1)


def make_all_forecast_clouds(clouds: np.array) -> np.array:
    for hour in tqdm(range(1, 6)):
        preds = make_current_hour_preds(clouds, hour, 'clouds')
        clouds = clouds.tolist()
        clouds.append([[preds[30 * row + col] for col in range(30)] for row in range(30)])
        clouds = np.array(clouds)
    return np.array([clouds[i] for i in range(43, 48)]).reshape(-1)


def make_all_forecast_humidity(humidity: np.array) -> np.array:
    for hour in tqdm(range(1, 6)):
        preds = make_current_hour_preds(humidity, hour, 'humidity')
        humidity = humidity.tolist()
        humidity.append([[preds[30 * row + col] for col in range(30)] for row in range(30)])
        humidity = np.array(humidity)
    return np.array([humidity[i] for i in range(43, 48)]).reshape(-1)


def make_all_forecast_pressure(pressure: np.array) -> np.array:
    for hour in tqdm(range(1, 6)):
        preds = make_current_hour_preds(pressure, hour, 'press')
        pressure = pressure.tolist()
        pressure.append([[preds[30 * row + col] for col in range(30)] for row in range(30)])
        pressure = np.array(pressure)
    return np.array([pressure[i] for i in range(43, 48)]).reshape(-1)


def make_all_forecast_temps(temps: np.array) -> np.array:
    for hour in tqdm(range(1, 6)):
        preds = make_current_hour_preds(temps, hour, 'temp')
        temps = temps.tolist()
        temps.append([[preds[30 * row + col] for col in range(30)] for row in range(30)])
        temps = np.array(temps)
    return np.array([temps[i] for i in range(43, 48)]).reshape(-1)


def make_current_hour_preds(arr: np.array, hour: int, pred_type: str) -> list[float]:
    preds = []
    for row in tqdm(range(30)):
        for col in range(30):
            n = get_neibrs(row, col)
            res = 0
            if pred_type == 'temp':
                df, future_df = create_df_temps(arr, n, hour, (row, col))
                res = prediction_temp(df, future_df, hour)
            elif pred_type == 'press':
                df, future_df = create_df_pressure(arr, n, hour, (row, col))
                res = prediction_pressure(df, future_df, hour)
            elif pred_type == 'humidity':
                df, future_df = create_df_humidity(arr, n, hour, (row, col))
                res = prediction_humidity(df, future_df, hour)
            elif pred_type == 'clouds':
                df, future_df = create_df_clouds(arr, n, hour, (row, col))
                res = prediction_clouds(df, future_df, hour)
            elif pred_type == 'speed':
                df, future_df = create_df_wind_speed(arr, n, hour, (row, col))
                res = prediction_speed(df, future_df, hour)
            preds.append(res)
    return preds


def prediction_pressure(df: pd.DataFrame, future_df: pd.DataFrame, hour: int) -> float:
    data_train = df.iloc[:43 + hour, :]
    mlf_forecast_humidity.fit(data_train, dropna=True, static_features=[])
    ans_df = mlf_forecast_humidity.predict(h=1, X_df=future_df.drop(columns=['y']))
    return ans_df.CatBoostRegressor.values[0]


def prediction_humidity(df: pd.DataFrame, future_df: pd.DataFrame, hour: int) -> float:
    data_train = df.iloc[:43 + hour, :]
    mlf_forecast_humidity.fit(data_train, dropna=True, static_features=[])
    ans_df = mlf_forecast_humidity.predict(h=1, X_df=future_df.drop(columns=['y']))
    return ans_df.CatBoostRegressor.values[0]


def prediction_speed(df: pd.DataFrame, future_df: pd.DataFrame, hour: int) -> float:
    data_train = df.iloc[:43 + hour, :]
    mlf_forecast_speed.fit(data_train, dropna=True, static_features=[])
    ans_df = mlf_forecast_speed.predict(h=1, X_df=future_df.drop(columns=['y']))
    return ans_df.XGBRegressor.values[0]

def prediction_temp(df: pd.DataFrame, future_df: pd.DataFrame, hour: int) -> float:
    data_train = df.iloc[:43 + hour, :]
    mlf_forecast_temps.fit(data_train, dropna=True, static_features=[])
    ans_df = mlf_forecast_temps.predict(h=1, X_df=future_df.drop(columns=['y']))
    return ans_df.CatBoostRegressor.values[0]


def prediction_clouds(df: pd.DataFrame, future_df: pd.DataFrame, hour: int) -> float:
    data_train = df.iloc[:43 + hour, :]
    mlf_forecast_clouds.fit(data_train, dropna=True, static_features=[])
    ans_df = mlf_forecast_clouds.predict(h=1, X_df=future_df.drop(columns=['y']))
    return ans_df.CatBoostRegressor.values[0]
