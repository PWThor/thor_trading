# features/config.py

# Technical indicator parameters
TECHNICAL_FEATURES = {
    'moving_averages': [5, 10, 20, 30, 90],
    'std_windows': [20, 90],
    'momentum_periods': [1, 5, 10, 20],
    'volatility_window': 20,
    'z_score_window': 90
}

# Seasonal feature parameters
SEASONAL_FEATURES = {
    'use_month': True,
    'use_day_of_year': True,
    'use_weekday': True,
    'winter_months': [11, 12, 1, 2, 3]
}

# Weather feature parameters
WEATHER_FEATURES = {
    'base_temperature': 65,  # For heating degree days
    'key_locations': ['New York', 'Chicago', 'Houston', 'Cushing', 'Boston']
}

# Target variable configuration
TARGET_CONFIG = {
    'horizon': 1,  # Predict next day by default
    'use_direction': True,
    'use_magnitude': True
}