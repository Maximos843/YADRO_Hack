from mlforecast import MLForecast
from window_ops.expanding import expanding_mean
from mlforecast.lag_transforms import RollingMean, ExponentiallyWeightedMean
from mlforecast.target_transforms import LocalStandardScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from config import Models



mlf_forecast_pressure = MLForecast(
    models=[CatBoostRegressor(**Models.PRESSURE_PARAMS)],
    freq=1,
    lags=[1, 2, 3, 4],
    lag_transforms = {
        1:  [expanding_mean],
        2: [expanding_mean],
    },
)
mlf_forecast_clouds = MLForecast(
    models={'model1': CatBoostRegressor(**Models.CLOUDS_PARAMS)},
    freq=1,
    lags=list(range(1, 5)),
    lag_transforms = {
        1:  [expanding_mean],
        2: [expanding_mean, RollingMean(window_size=3)],
        3: [expanding_mean, RollingMean(window_size=3)],
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