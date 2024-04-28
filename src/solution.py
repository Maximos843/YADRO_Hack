import numpy as np
import pandas as pd
from prediction import (make_all_forecast_clouds, make_all_forecast_humidity,
                        make_all_forecast_pressure, make_all_forecast_speed, make_all_forecast_temps)


if __name__ == '__main__':
    tempreture = np.load('/app/data/temperature.npy')
    clouds = np.load('/app/data/cloud_cover.npy')
    humidity = np.load('/app/data/humidity.npy')
    elevation = np.load('/app/data/elevation.npy')
    pressure = np.load('/app/data/pressure.npy')
    wind_dir = np.load('/app/data/wind_dir.npy')
    wind_speed = np.load('/app/data/wind_speed.npy')

    print('Speed Prediction')
    wind_speed_forecast = make_all_forecast_speed(wind_speed)
    print('Clouds Prediction')
    clouds_forecast = make_all_forecast_clouds(clouds)
    print('Humidity Prediction')
    humidity_forecast = make_all_forecast_clouds(humidity)
    print('Tempreture Prediction')
    tempreture_forecast = make_all_forecast_temps(tempreture)
    print('Pressure Prediction')
    pressure_forecast = make_all_forecast_pressure(pressure)
    wind_dir_forecast = np.array([14 for _ in range(4500)])

    solution = np.stack([
        tempreture_forecast.reshape(-1),
        pressure_forecast.reshape(-1),
        humidity_forecast.reshape(-1),
        wind_speed_forecast.reshape(-1),
        wind_dir_forecast.reshape(-1),
        clouds_forecast.reshape(-1),
    ], axis=1)
    solution = pd.DataFrame(solution, columns=['temperature', 'pressure', 'humidity', 'wind_speed', 'wind_dir', 'cloud_cover'])
    solution.to_csv('/app/solution/solution.csv', index_label='ID')
