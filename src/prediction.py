import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import (create_df_clouds, create_df_humidity, create_df_temps, create_df_wind_speed,
                   get_nine_neibrs, create_df_pressure, get_twenty_four_closest_neighbors)
from models import (mlf_forecast_pressure, mlf_forecast_clouds,
                           mlf_forecast_humidity, mlf_forecast_speed, mlf_forecast_temps)


def make_all_forecast(ans: np.array, pred_type: str) -> np.array:
    for hour in tqdm(range(1, 6)):
        preds = make_current_hour_preds(ans, hour, pred_type)
        ans = ans.tolist()
        ans.append([[preds[30 * row + col] for col in range(30)] for row in range(30)])
        ans = np.array(ans)
    return np.array([ans[i] for i in range(43, 48)]).reshape(-1)


def make_current_hour_preds(arr: np.array, hour: int, pred_type: str) -> list[float]:
    preds = []
    for row in tqdm(range(30)):
        for col in range(30):
            n = get_nine_neibrs(row, col)
            res = 0
            if pred_type == 'tempreture':
                df, future_df = create_df_temps(arr, n, hour, (row, col))
                res = prediction_temp(df, future_df, hour)
            elif pred_type == 'pressure':
                df, future_df = create_df_pressure(arr, n, hour, (row, col))
                res = prediction_pressure(df, future_df, hour)
            elif pred_type == 'humidity':
                df, future_df = create_df_humidity(arr, n, hour, (row, col))
                res = prediction_humidity(df, future_df, hour)
            elif pred_type == 'clouds':
                n = get_twenty_four_closest_neighbors((row, col))
                df, future_df = create_df_clouds(arr, n, hour, (row, col))
                res = prediction_clouds(df, future_df, hour, (row, col))
            elif pred_type == 'speed':
                df, future_df = create_df_wind_speed(arr, n, hour, (row, col))
                res = prediction_speed(df, future_df, hour)
            preds.append(res)
    return preds


def prediction_pressure(df: pd.DataFrame, future_df: pd.DataFrame, hour: int) -> float:
    data_train = df.iloc[:43 + hour, :]
    mlf_forecast_pressure.fit(data_train, dropna=True, static_features=[])
    ans_df = mlf_forecast_pressure.predict(h=1, X_df=future_df.drop(columns=['y']))
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


def prediction_clouds(df: pd.DataFrame, future_df: pd.DataFrame, hour: int, coord: tuple) -> float:
    data_train = df.iloc[:43 + hour, :]
    if coord[0] == 0 and coord[1] == 0:
        mlf_forecast_clouds.fit(data_train, dropna=True, static_features=[])
    ans_df = mlf_forecast_clouds.predict(h=1, X_df=future_df.drop(columns=['y']))
    return ans_df.model1.values[0]
