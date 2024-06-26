from dataclasses import dataclass


@dataclass
class Models:
    TEMPRETURE_PARAMS = {'iterations': 800, 'max_depth': 5, 'verbose': False, 'random_state': 3}
    CLOUDS_PARAMS = {'iterations': 1000, 'max_depth': 5, 'verbose': False, 'random_state': 3,
                     'l2_leaf_reg': 2, 'loss_function': 'Quantile:alpha=0.1'}
    HUMIDITY_PARAMS = {'iterations': 1000, 'max_depth': 5, 'verbose': False, 'random_state': 3}
    WIND_SPEED_PARAMS = {'max_depth': 4, 'random_state': 3}
    PRESSURE_PARAMS = {'iterations': 800, 'max_depth': 5, 'verbose': False, 'random_state': 3}
