# config.py
"""
Configuration for peak prediction analysis
"""

# Data source configuration - these are the QuerySpec parameters
DATA_SOURCES = {
    "load_mw": {
        "ms": "Eversource",
        "mn": "WMECo.estimated",
        "mp": "realtime_hourly_demand"
    },
    "forecast_3day_mw": {
        "ms": "Eversource",
        "mn": "WMECo.estimated",
        "mp": "three_day_load_forecast_mw"
    },
    "init_peak_mw": {
        "ms": "Eversource",
        "mn": "WMECo.peak_detect.v2",
        "mp": "init_peak_mw"
    }
}

# Peak prediction parameters
PEAK_PREDICT_PARAMS = {
    "zone": "Eversource",
    "threshold_mw": 50,
    "derate_7day_forecast": 0.97
}

# Initial peak finder parameters
INIT_PEAK_FINDER_PARAMS = {
    "default_alpha": 0.1,
    "lookback_years": 3
}

# Timezone
TIMEZONE = "America/New_York"

# Output configuration
OUTPUT_DIR = "./output"
OUTPUT_CSV_PEAK_PREDICT = "peak_predictions.csv"
OUTPUT_CSV_INIT_PEAK = "init_peak_finder.csv"
