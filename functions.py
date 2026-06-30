import os
import numpy as np
import polars as pl
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import xgboost as xgb
from datetime import datetime, timedelta
import calendar


#######################################################################################################################
# Feels Like Temperature Calculation #
#######################################################################################################################

def calculate_feels_like(temp_f, dewpoint_f):
    """
    Simple feels-like temperature based on humidity.

    Parameters:
    -----------
    temp_f : float
        Temperature in Fahrenheit
    dewpoint_f : float
        Dewpoint in Fahrenheit

    Returns:
    --------
    float
        Feels-like temperature in Fahrenheit
    """
    if temp_f is None or dewpoint_f is None:
        return temp_f if temp_f is not None else 70.0

    # Calculate relative humidity
    def vapor_pressure(t):
        return 6.112 * np.exp((17.67 * t) / (t + 243.5))

    try:
        vp_temp = vapor_pressure((temp_f - 32) * 5 / 9)
        vp_dewpoint = vapor_pressure((dewpoint_f - 32) * 5 / 9)
        rh = 100 * (vp_dewpoint / vp_temp)
        rh = np.clip(rh, 0, 100)
    except:
        rh = 50.0

    # Simple adjustment: high humidity makes it feel hotter, low humidity makes it feel cooler
    humidity_factor = (rh - 50) * 0.1  # ±5°F adjustment based on humidity
    feels_like = temp_f + humidity_factor

    return float(feels_like)


#######################################################################################################################
# Data Processing #
#######################################################################################################################

def process_zone_data_with_dewpoint(demand_df, temp_df, dewpoint_df, zone_name):
    """
    Process demand, temperature, and dewpoint data for a specific zone.
    Calculate heat index.
    Convert from UTC to America/New_York
    """
    # Standardize demand column names
    demand_df = demand_df.rename({demand_df.columns[1]: "value"})
    demand_df = demand_df.with_columns(pl.lit(zone_name).alias("zone"))

    # Standardize temperature column names
    temp_df = temp_df.rename({temp_df.columns[1]: "temperature"})

    # Standardize dewpoint column names
    dewpoint_df = dewpoint_df.rename({dewpoint_df.columns[1]: "dewpoint"})

    timezone = True

    if timezone:

        # Standardize timestamps and convert from UTC to Eastern Time
        demand_col = demand_df.columns[0]
        demand_df = demand_df.with_columns(
            pl.col(demand_col).cast(pl.Datetime("ms", "UTC")).dt.convert_time_zone("America/New_York").alias("timestamp")
        )
        if demand_col != "timestamp":
            demand_df = demand_df.drop(demand_col)

        temp_col = temp_df.columns[0]
        temp_df = temp_df.with_columns(
            pl.col(temp_col).cast(pl.Datetime("ms", "UTC")).dt.convert_time_zone("America/New_York").alias("timestamp")
        )
        if temp_col != "timestamp":
            temp_df = temp_df.drop(temp_col)

        dewpoint_col = dewpoint_df.columns[0]
        dewpoint_df = dewpoint_df.with_columns(
            pl.col(dewpoint_col).cast(pl.Datetime("ms", "UTC")).dt.convert_time_zone("America/New_York").alias("timestamp")
        )
        if dewpoint_col != "timestamp":
            dewpoint_df = dewpoint_df.drop(dewpoint_col)

    else:

        # Standardize dewpoint column names
        dewpoint_df = dewpoint_df.rename({dewpoint_df.columns[1]: "dewpoint"})

        # Standardize timestamps (no timezone conversion)
        demand_col = demand_df.columns[0]
        demand_df = demand_df.with_columns(
            pl.col(demand_col).cast(pl.Datetime).alias("timestamp")
        )
        if demand_col != "timestamp":
            demand_df = demand_df.drop(demand_col)

        temp_col = temp_df.columns[0]
        temp_df = temp_df.with_columns(
            pl.col(temp_col).cast(pl.Datetime).alias("timestamp")
        )
        if temp_col != "timestamp":
            temp_df = temp_df.drop(temp_col)

        dewpoint_col = dewpoint_df.columns[0]
        dewpoint_df = dewpoint_df.with_columns(
            pl.col(dewpoint_col).cast(pl.Datetime).alias("timestamp")
        )
        if dewpoint_col != "timestamp":
            dewpoint_df = dewpoint_df.drop(dewpoint_col)


    # Join demand with temperature and dewpoint
    zone_data = demand_df.join(
        temp_df.select(["timestamp", "temperature"]),
        on="timestamp",
        how="left"
    ).join(
        dewpoint_df.select(["timestamp", "dewpoint"]),
        on="timestamp",
        how="left"
    )

    # Calculate feels-like temperature
    zone_data = zone_data.with_columns(
        pl.struct(["temperature", "dewpoint"]).map_elements(
            lambda x: calculate_feels_like(x["temperature"], x["dewpoint"]),
            return_dtype=pl.Float64
        ).alias("feels_like")
    )

    return zone_data


#######################################################################################################################
# Percentile Calculations #
#######################################################################################################################

def calculate_percentiles_months(df, year_months, zones=None, percentiles=None):
    """
    Calculate percentiles for load data by zone and month, return as DataFrame.

    Parameters:
    -----------
    df : pl.DataFrame
        Polars DataFrame with columns: timestamp, value, zone, year_month
    year_months : list
        Sorted list of year-month strings (e.g., ["2024-06", "2024-07"])
    zones : list, optional
        List of zone names. Default: ["Connecticut", "Western Mass", "New Hampshire", "Total (NH+CT+WMass)"]
    percentiles : list, optional
        List of percentiles to calculate. Default: [10, 25, 75, 90, 95, 97.5, 99]

    Returns:
    --------
    results_df : pl.DataFrame
        DataFrame with columns: month, zone, p10, p25, p75, p90, p95, p97_5, p99
    """

    if zones is None:
        zones = ["Connecticut", "Western Mass", "New Hampshire", "Total (NH+CT+WMass)"]

    if percentiles is None:
        percentiles = [10, 25, 75, 90, 95, 97.5, 99]

    data = []

    # Extract unique months (01-12) from all year_months
    df_with_month = df.with_columns(
        pl.col("year_month").str.split("-").list.get(1).alias("month")
    )

    unique_months = sorted(df_with_month.select("month").unique().to_series().to_list())

    # For each month, calculate percentiles across all years
    for month in unique_months:
        month_data = df_with_month.filter(pl.col("month") == month)

        for zone in zones:
            zone_data = month_data.filter(pl.col("zone") == zone).select("value").to_series()

            # Remove NaN and infinite values
            zone_data = zone_data.drop_nulls()
            zone_data_np = zone_data.to_numpy()
            zone_data_np = zone_data_np[np.isfinite(zone_data_np)]

            if len(zone_data_np) > 0:
                row = {"month": month, "zone": zone}
                for percentile in percentiles:
                    col_name = f"p{percentile}".replace(".", "_")
                    row[col_name] = np.percentile(zone_data_np, percentile)
                data.append(row)

    results_df = pl.DataFrame(data)
    return results_df


def calculate_percentiles_years(df, year_months, zones=None, percentiles=None):
    """
    Calculate percentiles for load data by zone, year, and month, return as DataFrame.

    Parameters:
    -----------
    df : pl.DataFrame
        Polars DataFrame with columns: timestamp, value, zone, year_month
    year_months : list
        Sorted list of year-month strings (e.g., ["2024-06", "2024-07"])
    zones : list, optional
        List of zone names. Default: ["Connecticut", "Western Mass", "New Hampshire", "Total (NH+CT+WMass)"]
    percentiles : list, optional
        List of percentiles to calculate. Default: [10, 25, 75, 90, 95, 97.5, 99]

    Returns:
    --------
    results_df : pl.DataFrame
        DataFrame with columns: year, month, zone, p10, p25, p75, p90, p95, p97_5, p99
    """

    if zones is None:
        zones = ["Connecticut", "Western Mass", "New Hampshire", "Total (NH+CT+WMass)"]

    if percentiles is None:
        percentiles = [10, 25, 75, 90, 95, 97.5, 99]

    data = []

    # Extract year and month from year_month
    df_with_year_month = df.with_columns(
        pl.col("year_month").str.split("-").list.get(0).alias("year"),
        pl.col("year_month").str.split("-").list.get(1).alias("month")
    )

    unique_months = sorted(df_with_year_month.select("month").unique().to_series().to_list())
    unique_years = sorted(df_with_year_month.select("year").unique().to_series().to_list())

    # For each year and month combination, calculate percentiles
    for year in unique_years:
        for month in unique_months:
            month_data = df_with_year_month.filter(
                (pl.col("year") == year) & (pl.col("month") == month)
            )

            for zone in zones:
                zone_data = month_data.filter(pl.col("zone") == zone).select("value").to_series()

                # Remove NaN and infinite values
                zone_data = zone_data.drop_nulls()
                zone_data_np = zone_data.to_numpy()
                zone_data_np = zone_data_np[np.isfinite(zone_data_np)]

                if len(zone_data_np) > 0:
                    row = {"year": year, "month": month, "zone": zone}
                    for percentile in percentiles:
                        col_name = f"p{percentile}".replace(".", "_")
                        row[col_name] = np.percentile(zone_data_np, percentile)
                    data.append(row)

    results_df = pl.DataFrame(data)
    return results_df


def aggregate_percentiles_by_month(results_df):
    """
    Calculate average and median percentile values across all years for each month, zone, and percentile.

    Parameters:
    -----------
    results_df : pl.DataFrame
        DataFrame with columns: year, month, zone, p10, p25, p75, p90, p95, p97_5, p99

    Returns:
    --------
    aggregated_df : pl.DataFrame
        DataFrame with columns: month, zone, percentile, average, median
        One row per month-zone-percentile combination
    """

    # Get all percentile columns (p10, p25, p75, p90, p95, p97_5, p99)
    percentile_cols = [col for col in results_df.columns if col.startswith("p")]

    data = []

    # Get unique months and zones
    months = sorted(results_df.select("month").unique().to_series().to_list())
    zones = results_df.select("zone").unique().to_series().to_list()

    for month in months:
        for zone in zones:
            # Filter for specific month and zone
            month_zone_data = results_df.filter(
                (pl.col("month") == month) & (pl.col("zone") == zone)
            )

            for percentile_col in percentile_cols:
                values = month_zone_data.select(percentile_col).to_series()

                # Calculate average and median
                avg = values.mean()
                median = values.median()

                # Convert column name back to percentile label (p90 -> 90, p97_5 -> 97.5)
                percentile_label = percentile_col[1:].replace("_", ".")

                data.append({
                    "month": month,
                    "zone": zone,
                    "percentile": percentile_label,
                    "average": avg,
                    "median": median
                })

    aggregated_df = pl.DataFrame(data)
    return aggregated_df


def get_monthly_avg_of_daily_extremes(df, year_months):
    """
    Calculate monthly average of daily min/max values
    """
    # Find timestamp column
    timestamp_col = "timestamp"

    # Ensure timestamp is datetime
    df_with_dates = df.with_columns(
        pl.col(timestamp_col).cast(pl.Datetime).alias(timestamp_col)
    )

    # Extract date components
    df_with_dates = df_with_dates.with_columns([
        pl.col(timestamp_col).dt.year().alias("year"),
        pl.col(timestamp_col).dt.month().alias("month"),
        pl.col(timestamp_col).dt.date().alias("date")
    ])

    # Get daily min/max by zone
    daily_extremes = df_with_dates.group_by("zone", "year", "month", "date").agg([
        pl.col("value").min().alias("daily_min"),
        pl.col("value").max().alias("daily_max"),
    ])

    # Get monthly average of daily extremes
    monthly_stats = daily_extremes.group_by("zone", "year", "month").agg([
        pl.col("daily_min").mean().alias("avg_daily_min"),
        pl.col("daily_max").mean().alias("avg_daily_max"),
        pl.col("daily_min").min().alias("min_of_mins"),
        pl.col("daily_max").max().alias("max_of_maxs"),
    ])

    return monthly_stats


#######################################################################################################################
# Feature Engineering #
#######################################################################################################################

def engineer_features_with_weather_and_percentiles(df_input, percentile_months=None, monthly_stats=None):
    """
    Engineer features for the predictive model including:
    - ZONE-SPECIFIC temperature and feels-like
    - Percentile information for weighting
    - Historical monthly statistics
    - PEAK INDICATOR FEATURES
    """
    df_feat = df_input.clone()

    # Identify timestamp column
    timestamp_col = "timestamp"

    # Extract all temporal features
    df_feat = df_feat.with_columns([
        pl.col(timestamp_col).dt.year().alias("year"),
        pl.col(timestamp_col).dt.month().alias("month"),
        pl.col(timestamp_col).dt.day().alias("day"),
        pl.col(timestamp_col).dt.hour().alias("hour"),
        pl.col(timestamp_col).dt.weekday().alias("dayofweek"),
        pl.col(timestamp_col).dt.ordinal_day().alias("dayofyear"),
    ])

    # Add weekend indicator
    df_feat = df_feat.with_columns(
        is_weekend=pl.when(pl.col("dayofweek").is_in([5, 6])).then(1).otherwise(0)
    )

    # Lag features (by zone to maintain independence)
    df_feat = df_feat.with_columns([
        pl.col("value").shift(1).over("zone", "year", "month", "day").alias("value_lag_1h"),
        pl.col("value").shift(24).over("zone", "year", "month").alias("value_lag_24h"),
        pl.col("value").shift(168).over("zone").alias("value_lag_168h"),
    ])

    # Rolling averages for demand (by zone)
    df_feat = df_feat.with_columns([
        pl.col("value").rolling_mean(window_size=24).over("zone").alias("value_rolling_24h"),
        pl.col("value").rolling_mean(window_size=168).over("zone").alias("value_rolling_168h"),
    ])

    # ZONE-SPECIFIC Temperature lag features
    df_feat = df_feat.with_columns([
        pl.col("temperature").shift(1).over("zone").alias("temp_lag_1h"),
        pl.col("temperature").shift(24).over("zone").alias("temp_lag_24h"),
    ])

    # ZONE-SPECIFIC Temperature rolling averages
    df_feat = df_feat.with_columns([
        pl.col("temperature").rolling_mean(window_size=24).over("zone").alias("temp_rolling_24h"),
        pl.col("temperature").rolling_mean(window_size=168).over("zone").alias("temp_rolling_168h"),
    ])

    # ZONE-SPECIFIC Feels Like lag features
    df_feat = df_feat.with_columns([
        pl.col("feels_like").shift(1).over("zone").alias("feels_like_lag_1h"),
        pl.col("feels_like").shift(24).over("zone").alias("feels_like_lag_24h"),
    ])

    # ZONE-SPECIFIC Feels Like rolling averages
    df_feat = df_feat.with_columns([
        pl.col("feels_like").rolling_mean(window_size=24).over("zone").alias("feels_like_rolling_24h"),
        pl.col("feels_like").rolling_mean(window_size=168).over("zone").alias("feels_like_rolling_168h"),
    ])

    # Heating/Cooling degree days (base 65°F) - using zone-specific temps
    df_feat = df_feat.with_columns([
        pl.max_horizontal(65 - pl.col("temperature"), 0).alias("heating_degree_days"),
        pl.max_horizontal(pl.col("temperature") - 65, 0).alias("cooling_degree_days"),
    ])

    # Temperature bins - using zone-specific temps
    df_feat = df_feat.with_columns(
        pl.when(pl.col("temperature") < 20)
        .then(0)
        .when(pl.col("temperature") < 32)
        .then(1)
        .when(pl.col("temperature") < 50)
        .then(2)
        .when(pl.col("temperature") < 70)
        .then(3)
        .when(pl.col("temperature") < 85)
        .then(4)
        .when(pl.col("temperature") < 95)
        .then(5)
        .otherwise(6)
        .alias("temp_bin")
    )

    # Feels Like bins
    df_feat = df_feat.with_columns(
        pl.when(pl.col("feels_like") < 0)
        .then(0)
        .when(pl.col("feels_like") < 32)
        .then(1)
        .when(pl.col("feels_like") < 50)
        .then(2)
        .when(pl.col("feels_like") < 70)
        .then(3)
        .when(pl.col("feels_like") < 85)
        .then(4)
        .when(pl.col("feels_like") < 95)
        .then(5)
        .otherwise(6)
        .alias("feels_like_bin")
    )

    # Hour-of-day statistics (by zone)
    hourly_stats = df_feat.group_by("zone", "hour").agg([
        pl.col("value").mean().alias("hour_avg"),
        pl.col("value").std().alias("hour_std"),
        pl.col("value").quantile(0.90).alias("hour_p90"),
        pl.col("value").quantile(0.95).alias("hour_p95"),
    ])
    df_feat = df_feat.join(hourly_stats, on=["zone", "hour"], how="left")

    # Month-of-year statistics (by zone)
    monthly_stats_feat = df_feat.group_by("zone", "month").agg([
        pl.col("value").mean().alias("month_avg"),
        pl.col("value").std().alias("month_std"),
    ])
    df_feat = df_feat.join(monthly_stats_feat, on=["zone", "month"], how="left")

    # Day-of-week statistics (by zone)
    dow_stats = df_feat.group_by("zone", "dayofweek").agg([
        pl.col("value").mean().alias("dow_avg"),
        pl.col("value").std().alias("dow_std"),
    ])
    df_feat = df_feat.join(dow_stats, on=["zone", "dayofweek"], how="left")

    # ZONE-SPECIFIC Temperature statistics by hour
    temp_hourly_stats = df_feat.group_by("zone", "hour").agg([
        pl.col("temperature").mean().alias("temp_hour_avg"),
        pl.col("temperature").std().alias("temp_hour_std"),
    ])
    df_feat = df_feat.join(temp_hourly_stats, on=["zone", "hour"], how="left")

    # ZONE-SPECIFIC Temperature statistics by month
    temp_monthly_stats = df_feat.group_by("zone", "month").agg([
        pl.col("temperature").mean().alias("temp_month_avg"),
        pl.col("temperature").std().alias("temp_month_std"),
    ])
    df_feat = df_feat.join(temp_monthly_stats, on=["zone", "month"], how="left")

    # ZONE-SPECIFIC Feels Like statistics by hour
    fl_hourly_stats = df_feat.group_by("zone", "hour").agg([
        pl.col("feels_like").mean().alias("feels_like_hour_avg"),
        pl.col("feels_like").std().alias("feels_like_hour_std"),
    ])
    df_feat = df_feat.join(fl_hourly_stats, on=["zone", "hour"], how="left")

    # ZONE-SPECIFIC Feels Like statistics by month
    fl_monthly_stats = df_feat.group_by("zone", "month").agg([
        pl.col("feels_like").mean().alias("feels_like_month_avg"),
        pl.col("feels_like").std().alias("feels_like_month_std"),
    ])
    df_feat = df_feat.join(fl_monthly_stats, on=["zone", "month"], how="left")

    # Distance from peak for this hour
    df_feat = df_feat.with_columns([
        (pl.col("value") - pl.col("hour_p90")).alias("distance_from_p90"),
        (pl.col("value") - pl.col("hour_p95")).alias("distance_from_p95"),
        ((pl.col("value") >= pl.col("hour_p95")).cast(pl.Int32)).alias("is_peak_hour"),
        ((pl.col("value") >= pl.col("hour_p90")).cast(pl.Int32)).alias("is_high_hour"),
    ])

    # Add percentile information from percentile_months
    if percentile_months is not None:
        # Convert month to string in percentile_months to match df_feat
        percentile_months_prep = percentile_months.with_columns(
            pl.col("month").cast(pl.Utf8).str.zfill(2).alias("month")
        ).select(["month", "zone", "p90", "p95", "p97_5", "p99"])

        # Convert month to string in df_feat for the join
        df_feat_for_join = df_feat.with_columns(
            pl.col("month").cast(pl.Utf8).str.zfill(2).alias("month_str")
        )

        # Join on month_str and zone
        df_feat_for_join = df_feat_for_join.join(
            percentile_months_prep.rename({"month": "month_str"}),
            on=["month_str", "zone"],
            how="left"
        )

        # Drop the temporary month_str column and keep original month
        df_feat = df_feat_for_join.drop("month_str")
    else:
        # Create dummy percentile columns if not provided
        df_feat = df_feat.with_columns([
            pl.lit(0.0).alias("p90"),
            pl.lit(0.0).alias("p95"),
            pl.lit(0.0).alias("p97_5"),
            pl.lit(0.0).alias("p99"),
        ])

    # Fill NaN values in lag/rolling features with zone averages
    df_feat = df_feat.with_columns([
        pl.col("value_lag_1h").fill_null(pl.col("hour_avg")),
        pl.col("value_lag_24h").fill_null(pl.col("hour_avg")),
        pl.col("value_lag_168h").fill_null(pl.col("hour_avg")),
        pl.col("value_rolling_24h").fill_null(pl.col("hour_avg")),
        pl.col("value_rolling_168h").fill_null(pl.col("hour_avg")),
        pl.col("temp_lag_1h").fill_null(pl.col("temperature")),
        pl.col("temp_lag_24h").fill_null(pl.col("temperature")),
        pl.col("temp_rolling_24h").fill_null(pl.col("temperature")),
        pl.col("temp_rolling_168h").fill_null(pl.col("temperature")),
        pl.col("feels_like_lag_1h").fill_null(pl.col("feels_like")),
        pl.col("feels_like_lag_24h").fill_null(pl.col("feels_like")),
        pl.col("feels_like_rolling_24h").fill_null(pl.col("feels_like")),
        pl.col("feels_like_rolling_168h").fill_null(pl.col("feels_like")),
        pl.col("distance_from_p90").fill_null(0.0),
        pl.col("distance_from_p95").fill_null(0.0),
    ])

    return df_feat



#######################################################################################################################
# Model Training #
#######################################################################################################################

def train_zone_model_with_weather_and_percentiles(df_zone, zone_name, percentile_threshold=95, test_size=0.2):
    """
    Train XGBoost model with temperature, feels-like, and percentile-based sample weighting.

    Samples at or above the percentile_threshold are weighted more heavily during training.

    Parameters:
    -----------
    df_zone : pl.DataFrame
        Zone-specific feature dataframe
    zone_name : str
        Name of the zone
    percentile_threshold : float
        Percentile threshold for sample weighting (default: 95)
    test_size : float
        Proportion of data to use for testing

    Returns:
    --------
    dict
        Dictionary containing model, scalers, feature columns, and metrics
    """
    print(f"\nTraining model for {zone_name}...")

    for col in df_zone.columns:
        null_count = df_zone.select(col).null_count().item()
        if null_count > 0:
            print(f"    {col}: {null_count} nulls")

    # Feature columns - now includes feels_like, percentile features, AND PEAK FEATURES
    feature_cols = [
        # Temporal features
        'hour', 'dayofweek', 'is_weekend', 'month', 'day',

        # Demand lag features
        'value_lag_1h', 'value_lag_24h', 'value_lag_168h',
        'value_rolling_24h', 'value_rolling_168h',

        # Demand statistics
        'hour_avg', 'hour_std',
        'month_avg', 'month_std',
        'dow_avg', 'dow_std',

        # TEMPERATURE FEATURES
        'temperature',
        'temp_lag_1h', 'temp_lag_24h',
        'temp_rolling_24h', 'temp_rolling_168h',
        'heating_degree_days', 'cooling_degree_days',
        'temp_bin',
        'temp_hour_avg', 'temp_hour_std',
        'temp_month_avg', 'temp_month_std',

        # FEELS LIKE FEATURES
        'feels_like',
        'feels_like_lag_1h', 'feels_like_lag_24h',
        'feels_like_rolling_24h', 'feels_like_rolling_168h',
        'feels_like_bin',
        'feels_like_hour_avg', 'feels_like_hour_std',
        'feels_like_month_avg', 'feels_like_month_std',

        # PEAK INDICATOR FEATURES (NEW)
        'hour_p90', 'hour_p95',
        'distance_from_p90', 'distance_from_p95',
        'is_peak_hour', 'is_high_hour',

        # PERCENTILE FEATURES
        'p90', 'p95', 'p97_5', 'p99',
    ]

    # Filter available columns
    feature_cols = [col for col in feature_cols if col in df_zone.columns]

    print(f"  Feature columns: {len(feature_cols)}")
    print(f"  Available columns: {df_zone.columns}")

    # Convert to numpy
    X = df_zone.select(feature_cols).to_numpy()
    y = df_zone.select("value").to_numpy().flatten()

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")

    if X.shape[0] == 0:
        print(f"  ERROR: No data for {zone_name}!")
        return None

    # Calculate sample weights based on percentile threshold
    # Samples with values >= percentile_threshold get higher weight
    percentile_value = np.percentile(y, percentile_threshold)
    sample_weights = np.where(y >= percentile_value, 2.0, 1.0)  # 2x weight for high-demand samples

    print(f"  Percentile threshold ({percentile_threshold}th): {percentile_value:.0f} MW")
    print(f"  High-demand samples (weight=2.0): {np.sum(sample_weights > 1.0)} / {len(sample_weights)}")

    # Normalize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Train-test split (maintain time order)
    split_idx = int(len(X_scaled) * (1 - test_size))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
    weights_train = sample_weights[:split_idx]

    # Train model with sample weights
    model = XGBRegressor(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20
    )

    model.fit(
        X_train, y_train,
        sample_weight=weights_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluate
    y_pred_test = model.predict(X_test)
    y_pred_test_original = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_test_original))
    mae = mean_absolute_error(y_test_original, y_pred_test_original)
    r2 = r2_score(y_test_original, y_pred_test_original)

    # Calculate MAPE safely
    mask = y_test_original != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_test_original[mask] - y_pred_test_original[mask]) / y_test_original[mask])) * 100
    else:
        mape = np.nan

    print(f"  RMSE: {rmse:.2f} MW")
    print(f"  MAE: {mae:.2f} MW")
    print(f"  R² Score: {r2:.4f}")
    if not np.isnan(mape):
        print(f"  MAPE: {mape:.2f}%")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n  Top 10 Features for {zone_name}:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"    - {row['feature']}: {row['importance']:.4f}")

    return {
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_cols': feature_cols,
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape if not np.isnan(mape) else None
        },
        'feature_importance': feature_importance,
        'percentile_threshold': percentile_threshold,
        'percentile_value': percentile_value
    }



#######################################################################################################################
# Prediction #
#######################################################################################################################

def predict_month_with_percentiles_and_weather(zone, month, year_to_predict, df_input, model_dict,
                                               current_weather_data=None, percentile_data=None,
                                               monthly_stats=None):
    """
    Predict all hours for a specific month and zone with:
    - Dynamic temperature and feels-like adjustment
    - Percentile-based bounds
    - Historical monthly statistics

    Parameters:
    -----------
    zone : str
        Zone name
    month : int
        Month number (1-12)
    year_to_predict : int
        Year to predict
    df_input : pl.DataFrame
        Input dataframe with historical data
    model_dict : dict
        Dictionary containing model, scalers, and feature columns
    current_weather_data : pl.DataFrame, optional
        Current/forecast weather data with temperature, dewpoint, feels_like
    percentile_data : pl.DataFrame, optional
        Percentile data for bounds
    monthly_stats : pl.DataFrame, optional
        Monthly statistics for historical context

    Returns:
    --------
    pl.DataFrame
        Predictions dataframe
    """

    model = model_dict['model']
    scaler_X = model_dict['scaler_X']
    scaler_y = model_dict['scaler_y']
    feature_cols = model_dict['feature_cols']

    # Get historical data for that month and zone
    historical = df_input.filter(
        (pl.col("zone") == zone) &
        (pl.col("month") == month)
    )

    if historical.shape[0] == 0:
        print(f"Warning: No historical data for {zone} in month {month}")
        return None

    predictions = []

    import calendar
    days_in_month = calendar.monthrange(year_to_predict, month)[1]

    # Calculate temperature and feels-like adjustment factors
    temp_adjustment_factor = 1.0
    fl_adjustment_factor = 1.0
    current_avg_temp = None
    current_avg_fl = None
    historical_avg_temp = None
    historical_avg_fl = None

    if current_weather_data is not None and current_weather_data.shape[0] > 0:
        # Get average of current temperatures and feels-like
        current_avg_temp = current_weather_data.select("temperature").mean().item()
        current_avg_fl = current_weather_data.select("feels_like").mean().item()

        # Get historical averages for this month
        historical_avg_temp = historical.select("temperature").mean().item()
        historical_avg_fl = historical.select("feels_like").mean().item()

        if historical_avg_temp is not None and historical_avg_temp != 0:
            temp_adjustment_factor = current_avg_temp / historical_avg_temp

        if historical_avg_fl is not None and historical_avg_fl != 0:
            fl_adjustment_factor = current_avg_fl / historical_avg_fl

        print(f"  Weather adjustment for {zone}:")
        print(
            f"    Temperature: {historical_avg_temp:.1f}°F → {current_avg_temp:.1f}°F (factor: {temp_adjustment_factor:.3f})")
        print(
            f"    Feels Like: {historical_avg_fl:.1f}°F → {current_avg_fl:.1f}°F (factor: {fl_adjustment_factor:.3f})")

    for day in range(1, days_in_month + 1):
        for hour in range(24):
            try:
                # Get historical data for this hour
                hour_data = historical.filter(pl.col("hour") == hour)

                # Calculate day of week for prediction date
                pred_date = datetime(year_to_predict, month, day)
                dow = pred_date.weekday()
                is_weekend = 1 if dow in [5, 6] else 0

                # Get temperature and feels-like with dynamic adjustment
                forecast_temp = None
                forecast_fl = None
                temp_source = "historical"

                # First, try to get actual current temperature if available
                if current_weather_data is not None:
                    weather_match = current_weather_data.filter(
                        (pl.col("timestamp").dt.date() == pred_date.date()) &
                        (pl.col("timestamp").dt.hour() == hour)
                    )

                    if weather_match.shape[0] > 0:
                        forecast_temp = weather_match.select("temperature").item()
                        forecast_fl = weather_match.select("feels_like").item()
                        temp_source = "current_actual"

                # If no current data, use historical average with adjustment
                if forecast_temp is None:
                    historical_temp = hour_data.select("temperature").mean().item() or 70
                    historical_fl = hour_data.select("feels_like").mean().item() or 70

                    # Apply adjustment factors to historical values
                    if temp_adjustment_factor != 1.0:
                        forecast_temp = historical_temp * temp_adjustment_factor
                        forecast_fl = historical_fl * fl_adjustment_factor
                        temp_source = "historical_adjusted"
                    else:
                        forecast_temp = historical_temp
                        forecast_fl = historical_fl
                        temp_source = "historical_baseline"

                # Calculate temp_bin with NEW RANGES (0-6)
                if forecast_temp < 0:
                    temp_bin = 0
                elif forecast_temp < 20:
                    temp_bin = 1
                elif forecast_temp < 50:
                    temp_bin = 2
                elif forecast_temp < 70:
                    temp_bin = 3
                elif forecast_temp < 85:
                    temp_bin = 4
                elif forecast_temp < 95:
                    temp_bin = 5
                else:
                    temp_bin = 6

                # Calculate feels_like_bin
                if forecast_fl < 0:
                    feels_like_bin = 0
                elif forecast_fl < 32:
                    feels_like_bin = 1
                elif forecast_fl < 50:
                    feels_like_bin = 2
                elif forecast_fl < 70:
                    feels_like_bin = 3
                elif forecast_fl < 85:
                    feels_like_bin = 4
                elif forecast_fl < 95:
                    feels_like_bin = 5
                else:
                    feels_like_bin = 6

                # Get percentile values for this month and zone
                p90 = 0.0
                p95 = 0.0
                p97_5 = 0.0
                p99 = 0.0

                if percentile_data is not None:
                    percentile_match = percentile_data.filter(
                        (pl.col("month") == str(month).zfill(2)) &
                        (pl.col("zone") == zone)
                    )
                    if percentile_match.shape[0] > 0:
                        p90 = percentile_match.select("p90").item() or 0.0
                        p95 = percentile_match.select("p95").item() or 0.0
                        p97_5 = percentile_match.select("p97_5").item() or 0.0
                        p99 = percentile_match.select("p99").item() or 0.0

                # Create feature vector
                features_dict = {
                    'hour': float(hour),
                    'dayofweek': float(dow),
                    'is_weekend': float(is_weekend),
                    'month': float(month),
                    'day': float(day),
                    'value_lag_1h': float(
                        hour_data.select("value_lag_1h").mean().item() or historical.select("value").mean().item()),
                    'value_lag_24h': float(
                        hour_data.select("value_lag_24h").mean().item() or historical.select("value").mean().item()),
                    'value_lag_168h': float(
                        hour_data.select("value_lag_168h").mean().item() or historical.select("value").mean().item()),
                    'value_rolling_24h': float(hour_data.select("value_rolling_24h").mean().item() or historical.select(
                        "value").mean().item()),
                    'value_rolling_168h': float(
                        hour_data.select("value_rolling_168h").mean().item() or historical.select(
                            "value").mean().item()),
                    'hour_avg': float(
                        hour_data.select("hour_avg").mean().item() or historical.select("value").mean().item()),
                    'hour_std': float(hour_data.select("hour_std").mean().item() or 0),
                    'month_avg': float(historical.select("month_avg").mean().item()),
                    'month_std': float(historical.select("month_std").mean().item()),
                    'dow_avg': float(
                        hour_data.select("dow_avg").mean().item() or historical.select("value").mean().item()),
                    'dow_std': float(hour_data.select("dow_std").mean().item() or 0),
                    'temperature': float(forecast_temp),
                    'temp_lag_1h': float(forecast_temp),
                    'temp_lag_24h': float(forecast_temp),
                    'temp_rolling_24h': float(forecast_temp),
                    'temp_rolling_168h': float(forecast_temp),
                    'heating_degree_days': float(max(0, 65 - forecast_temp)),
                    'cooling_degree_days': float(max(0, forecast_temp - 65)),
                    'temp_bin': float(temp_bin),
                    'temp_hour_avg': float(hour_data.select("temp_hour_avg").mean().item() or forecast_temp),
                    'temp_hour_std': float(hour_data.select("temp_hour_std").mean().item() or 0),
                    'temp_month_avg': float(historical.select("temp_month_avg").mean().item()),
                    'temp_month_std': float(historical.select("temp_month_std").mean().item()),
                    'feels_like': float(forecast_fl),
                    'feels_like_lag_1h': float(forecast_fl),
                    'feels_like_lag_24h': float(forecast_fl),
                    'feels_like_rolling_24h': float(forecast_fl),
                    'feels_like_rolling_168h': float(forecast_fl),
                    'feels_like_bin': float(feels_like_bin),
                    'feels_like_hour_avg': float(hour_data.select("feels_like_hour_avg").mean().item() or forecast_fl),
                    'feels_like_hour_std': float(hour_data.select("feels_like_hour_std").mean().item() or 0),
                    'feels_like_month_avg': float(historical.select("feels_like_month_avg").mean().item()),
                    'feels_like_month_std': float(historical.select("feels_like_month_std").mean().item()),
                    'p90': float(p90),
                    'p95': float(p95),
                    'p97_5': float(p97_5),
                    'p99': float(p99),
                }

                # Build feature array in correct order
                features = np.array([[features_dict.get(col, 0.0) for col in feature_cols]])

                # Scale and predict
                features_scaled = scaler_X.transform(features)
                pred_scaled = model.predict(features_scaled)[0]
                pred_original = scaler_y.inverse_transform([[pred_scaled]])[0][0]

                # Get percentile bounds from historical data
                hist_values = hour_data.select("value").to_numpy().flatten()

                if len(hist_values) > 0:
                    p10 = float(np.percentile(hist_values, 10))
                    p25 = float(np.percentile(hist_values, 25))
                    p75 = float(np.percentile(hist_values, 75))
                    p90_hist = float(np.percentile(hist_values, 90))
                    hist_min = float(np.min(hist_values))
                    hist_max = float(np.max(hist_values))
                    hist_avg = float(np.mean(hist_values))
                else:
                    p10 = p25 = p75 = p90_hist = hist_min = hist_max = hist_avg = float(pred_original)

                predictions.append({
                    'timestamp': pred_date.replace(hour=hour),
                    'zone': zone,
                    'hour': hour,
                    'day': day,
                    'predicted_mw': float(pred_original),
                    'p10': p10,
                    'p25': p25,
                    'p75': p75,
                    'p90': p90_hist,
                    'historical_min': hist_min,
                    'historical_max': hist_max,
                    'historical_avg': hist_avg,
                    'temperature_used': float(forecast_temp),
                    'feels_like_used': float(forecast_fl),
                    'temp_source': temp_source,
                    'temp_adjustment_factor': float(temp_adjustment_factor),
                })
            except Exception as e:
                continue

    if len(predictions) == 0:
        print(f"Warning: No predictions generated for {zone} in month {month}")
        return None

    return pl.DataFrame(predictions)


def save_predictions(predictions_list, target_year, output_dir="predict/data/"):
    """
    Combine and save predictions to CSV

    Parameters:
    -----------
    predictions_list : list
        List of prediction dataframes
    target_year : int
        Year being predicted
    output_dir : str
        Directory to save predictions

    Returns:
    --------
    pl.DataFrame or None
        Combined predictions dataframe or None if empty
    """

    os.makedirs(output_dir, exist_ok=True)

    if len(predictions_list) == 0:
        print("ERROR: No predictions were generated!")
        return None

    # Combine all predictions
    predictions_combined = pl.concat(predictions_list)

    print(f"\nTotal predictions generated: {predictions_combined.shape[0]}")

    # Save predictions
    filepath = os.path.join(output_dir, f"predictions_{target_year}.csv")
    predictions_combined.write_csv(filepath)
    print(f"\nPredictions saved to: {filepath}")

    return predictions_combined


def train_zone_model_quantile(df_zone, zone_name, quantile=0.90, test_size=0.2):
    """
    Train XGBoost model using quantile regression to predict high-load scenarios.

    Instead of predicting the mean, this predicts the 90th percentile load.
    This naturally focuses on peak scenarios.

    Parameters:
    -----------
    df_zone : pl.DataFrame
        Zone-specific feature dataframe
    zone_name : str
        Name of the zone
    quantile : float
        Quantile to predict (0.90 = 90th percentile, 0.95 = 95th percentile)
    test_size : float
        Proportion of data to use for testing

    Returns:
    --------
    dict
        Dictionary containing model, scalers, feature columns, and metrics
    """
    print(f"\nTraining {quantile:.0%} quantile model for {zone_name}...")

    for col in df_zone.columns:
        null_count = df_zone.select(col).null_count().item()
        if null_count > 0:
            print(f"    {col}: {null_count} nulls")

    # Feature columns
    feature_cols = [
        # Temporal features
        'hour', 'dayofweek', 'is_weekend', 'month', 'day',

        # Demand lag features
        'value_lag_1h', 'value_lag_24h', 'value_lag_168h',
        'value_rolling_24h', 'value_rolling_168h',

        # Demand statistics
        'hour_avg', 'hour_std',
        'month_avg', 'month_std',
        'dow_avg', 'dow_std',

        # TEMPERATURE FEATURES
        'temperature',
        'temp_lag_1h', 'temp_lag_24h',
        'temp_rolling_24h', 'temp_rolling_168h',
        'heating_degree_days', 'cooling_degree_days',
        'temp_bin',
        'temp_hour_avg', 'temp_hour_std',
        'temp_month_avg', 'temp_month_std',

        # FEELS LIKE FEATURES
        'feels_like',
        'feels_like_lag_1h', 'feels_like_lag_24h',
        'feels_like_rolling_24h', 'feels_like_rolling_168h',
        'feels_like_bin',
        'feels_like_hour_avg', 'feels_like_hour_std',
        'feels_like_month_avg', 'feels_like_month_std',

        # PEAK INDICATOR FEATURES
        'hour_p90', 'hour_p95',
        'distance_from_p90', 'distance_from_p95',
        'is_peak_hour', 'is_high_hour',

        # PERCENTILE FEATURES
        'p90', 'p95', 'p97_5', 'p99',
    ]

    # Filter available columns
    feature_cols = [col for col in feature_cols if col in df_zone.columns]

    print(f"  Feature columns: {len(feature_cols)}")

    # Convert to numpy
    X = df_zone.select(feature_cols).to_numpy()
    y = df_zone.select("value").to_numpy().flatten()

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")

    if X.shape[0] == 0:
        print(f"  ERROR: No data for {zone_name}!")
        return None

    # Normalize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Train-test split (maintain time order)
    split_idx = int(len(X_scaled) * (1 - test_size))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

    # Quantile regression - predicts the quantile, not the mean
    # Use reg:quantileerror instead of reg:quantilehubererror
    model = XGBRegressor(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:quantileerror',  # Changed from reg:quantilehubererror
        quantile_alpha=quantile,  # Predict this quantile
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluate
    y_pred_test = model.predict(X_test)
    y_pred_test_original = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_test_original))
    mae = mean_absolute_error(y_test_original, y_pred_test_original)
    r2 = r2_score(y_test_original, y_pred_test_original)

    print(f"  RMSE: {rmse:.2f} MW")
    print(f"  MAE: {mae:.2f} MW")
    print(f"  R² Score: {r2:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n  Top 10 Features for {zone_name}:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"    - {row['feature']}: {row['importance']:.4f}")

    return {
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_cols': feature_cols,
        'quantile': quantile,
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        },
        'feature_importance': feature_importance
    }


