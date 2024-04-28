import pandas as pd
import numpy as np


def create_df_pressure(pressure: np.array, n: list[tuple], hour: int, coord: tuple) -> tuple:
    df = pd.DataFrame(data=[['0/0', h + 1, pressure[h][coord[0]][coord[1]], pressure[h][n[0][0]][n[0][1]], pressure[h][n[1][0]][n[1][1]], pressure[h][n[2][0]][n[2][1]],
                            pressure[h][n[3][0]][n[3][1]], pressure[h][n[4][0]][n[4][1]], pressure[h][n[5][0]][n[5][1]], pressure[h][n[6][0]][n[6][1]],
                            pressure[h][n[7][0]][n[7][1]]]  for h in range(43 + hour - 1)],
            columns=['unique_id', 'ds', 'y', 'neibr1', 'neibr2', 'neibr3', 'neibr4', 'neibr5', 'neibr6', 'neibr7', 'neibr8'])
    future_df = df.iloc[[-1], :].copy()
    future_df['ds'] = 43 + hour
    for col in df.columns[3:]:
        df[col] = df[col].shift(1)
    return df, future_df


def create_df_wind_speed(wind_speed: np.array, n: list[tuple], hour: int, coord: tuple) -> tuple:
    df = pd.DataFrame(data=[['0/0', h + 1, wind_speed[h][coord[0]][coord[1]]] + [wind_speed[h][i[0]][i[1]] for i in n] + [wind_speed[h][i[0]][i[1]] for i in n]
                            + [wind_speed[h][i[0]][i[1]] for i in n] + [wind_speed[h][i[0]][i[1]] for i in n] + [wind_speed[h][i[0]][i[1]] for i in n]
                            + [wind_speed[h][i[0]][i[1]] for i in n] for h in range(43 + hour - 1)],
            columns=['unique_id', 'ds', 'y', 'neibr1_shift1', 'neibr2_shift1', 'neibr3_shift1', 'neibr4_shift1', 'neibr5_shift1', 'neibr6_shift1', 'neibr7_shift1', 'neibr8_shift1',
                     'neibr1_shift2', 'neibr2_shift2', 'neibr3_shift2', 'neibr4_shift2', 'neibr5_shift2', 'neibr6_shift2', 'neibr7_shift2', 'neibr8_shift2',
                     'neibr1_shift3', 'neibr2_shift3', 'neibr3_shift3', 'neibr4_shift3', 'neibr5_shift3', 'neibr6_shift3', 'neibr7_shift3', 'neibr8_shift3',
                     'neibr1_shift4', 'neibr2_shift4', 'neibr3_shift4', 'neibr4_shift4', 'neibr5_shift4', 'neibr6_shift4', 'neibr7_shift4', 'neibr8_shift4',
                     'neibr1_shift5', 'neibr2_shift5', 'neibr3_shift5', 'neibr4_shift5', 'neibr5_shift5', 'neibr6_shift5', 'neibr7_shift5', 'neibr8_shift5',
                     'neibr1_shift6', 'neibr2_shift6', 'neibr3_shift6', 'neibr4_shift6', 'neibr5_shift6', 'neibr6_shift6', 'neibr7_shift6', 'neibr8_shift6',])
    df['y'] = double_exponential_smoothing(df.y, 0.3, 0.14)
    future_df = df.iloc[[-1], :].copy()
    future_df['ds'] = 43 + hour
    for col in df.columns[3:11]:
        df[col] = df[col].shift(1)
    for col in df.columns[11:19]:
        df[col] = df[col].shift(2)
    for col in df.columns[19:27]:
        df[col] = df[col].shift(3)
    for col in df.columns[27:35]:
        df[col] = df[col].shift(4)
    for col in df.columns[43:51]:
        df[col] = df[col].shift(5)
    for col in df.columns[51:]:
        df[col] = df[col].shift(6)
    return df, future_df


def create_df_clouds(clouds: np.array, n: list[tuple], hour: int, coord: tuple) -> tuple:
    df = pd.DataFrame(data=[['0/0', h + 1] + [clouds[h][coord[0]][coord[1]]] + [clouds[h][i[0]][i[1]] for i in n] + [clouds[h][i[0]][i[1]] for i in n]
         + [clouds[h][i[0]][i[1]] for i in n] + [clouds[h][i[0]][i[1]] for i in n] for h in range(43 + hour - 1)],
                    columns=['unique_id', 'ds', 'y', 'neibr1_cloud', 'neibr2_cloud', 'neibr3_cloud', 'neibr4_cloud', 'neibr5_cloud', 'neibr6_cloud',
                            'neibr7_cloud', 'neibr8_cloud', 'neibr1_cloud_shift2', 'neibr2_cloud_shift2', 'neibr3_cloud_shift2', 'neibr4_cloud_shift2',
                             'neibr5_cloud_shift2', 'neibr6_cloud_shift2', 'neibr7_cloud_shift2', 'neibr8_cloud_shift2',
                             'neibr1_cloud_shift3', 'neibr2_cloud_shift3', 'neibr3_cloud_shift3', 'neibr4_cloud_shift3',
                             'neibr5_cloud_shift3', 'neibr6_cloud_shift3', 'neibr7_cloud_shift3', 'neibr8_cloud_shift3',
                             'neibr1_cloud_shift4', 'neibr2_cloud_shift4', 'neibr3_cloud_shift4', 'neibr4_cloud_shift4',
                             'neibr5_cloud_shift4', 'neibr6_cloud_shift4', 'neibr7_cloud_shift4', 'neibr8_cloud_shift4',])
    df['y'] = double_exponential_smoothing(df.y, 0.3, 0.2)
    future_df = df.iloc[[-1], :].copy()
    future_df['ds'] = 43 + hour
    for col in df.columns[3:11]:
        df[col] = df[col].shift(1)
    for col in df.columns[11:19]:
        df[col] = df[col].shift(2)
    for col in df.columns[19:27]:
        df[col] = df[col].shift(3)
    for col in df.columns[27:]:
        df[col] = df[col].shift(4)
    return df, future_df


def create_df_humidity(humidity: np.array, n: list[tuple], hour: int, coord: tuple) -> tuple:
    df = pd.DataFrame(data=[['0/0', h + 1, humidity[h][coord[0]][coord[1]]] + [humidity[h][i[0]][i[1]] for i in n] + [humidity[h][i[0]][i[1]] for i in n]  for h in range(43 + hour - 1)],
                    columns=['unique_id', 'ds', 'y', 'neibr1', 'neibr2', 'neibr3', 'neibr4', 'neibr5', 'neibr6', 'neibr7', 'neibr8',
                            'neibr1_shift2', 'neibr2_shift2', 'neibr3_shift2', 'neibr4_shift2', 'neibr5_shift2', 'neibr6_shift2', 'neibr7_shift2', 'neibr8_shift2'])
    df['y'] = double_exponential_smoothing(df.y, 0.4, 0.2)
    future_df = df.iloc[[-1], :].copy()
    future_df['ds'] = 43 + hour
    for col in df.columns[3:11]:
        df[col] = df[col].shift(1)
    for col in df.columns[11:]:
        df[col] = df[col].shift(2)
    return df, future_df


def create_df_temps(temps: np.array, n: list[tuple], hour: int, coord: tuple) -> tuple:
    df = pd.DataFrame(data=[['0/0', h + 1, temps[h][coord[0]][coord[1]]] + [temps[h][i[0]][i[1]] for i in n] + [temps[h][i[0]][i[1]] for i in n]  for h in range(43 + hour - 1)],
        columns=['unique_id', 'ds', 'y', 'neibr1', 'neibr2', 'neibr3', 'neibr4', 'neibr5', 'neibr6', 'neibr7', 'neibr8',
         'neibr1_shift2', 'neibr2_shift2', 'neibr3_shift2', 'neibr4_shift2', 'neibr5_shift2', 'neibr6_shift2', 'neibr7_shift2', 'neibr8_shift2'])
    df.y = double_exponential_smoothing(df.y, 0.4, 0.2)
    future_df = df.iloc[[-1], :].copy()
    future_df['ds'] = 43 + hour
    for col in df.columns[3:11]:
        df[col] = df[col].shift(1)
    for col in df.columns[11:19]:
        df[col] = df[col].shift(2)
    return df, future_df


def double_exponential_smoothing(series: pd.Series, alpha: float, beta: float) -> list:
    result = [series[0]]
    for n in range(1, len(series) + 1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series):
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)
    return result[:-1]


def get_neibrs(row: int, col: int) -> list[tuple]:
    if row == 0 and col == 0:
        n = [(0, 1), (1, 0), (1, 1), (2, 1), (2, 2), (1, 2), (2, 0), (0, 2)]
    elif row == 29 and col == 0:
        n = [(29, 1), (28, 0), (28, 1), (27, 0), (27, 1), (27, 2), (28, 2), (29, 2)]
    elif row == 0 and col == 29:
        n = [(0, 28), (1, 28), (1, 29), (0, 27), (1, 27), (2, 27), (2, 28), (2, 29)]
    elif row == 29 and col == 29:
        n = [(29, 28), (28, 28), (28, 29), (29, 27), (28, 27), (27, 27), (27, 28), (27, 29)]
    elif row == 0 and col != 29 and col != 0:
        n = [(row, col - 1), (row, col + 1), (row + 1, col - 1), (row + 1, col), (row + 1, col + 1), (row + 2, col - 1), (row + 2, col), (row + 2, col + 1)]
    elif row == 29 and col != 29 and col != 0:
        n = [(row, col - 1), (row, col + 1), (row - 1, col - 1), (row - 1, col), (row - 1, col + 1), (row - 2, col - 1), (row - 2, col), (row - 2, col + 1)]
    elif col == 0 and row != 29 and row != 0:
        n = [(row + 1, col), (row - 1, col), (row + 1, col + 1), (row, col + 1), (row - 1, col + 1), (row + 1, col + 2), (row, col + 2), (row - 1, col + 2)]
    elif col == 29 and row != 29 and row != 0:
        n = [(row + 1, col), (row - 1, col), (row + 1, col - 1), (row, col - 1), (row - 1, col - 1), (row + 1, col - 2), (row, col - 2), (row - 1, col - 2)]
    else:
        n = [(row - 1, col - 1), (row - 1, col), (row - 1, col + 1), (row, col - 1), (row, col + 1), (row + 1, col - 1), (row + 1, col), (row + 1, col + 1)]
    return n
