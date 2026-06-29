#######################################################################################################################
# Imports #
#######################################################################################################################
import sys
import polars as pl
import pandas as pd
import numpy as np
import inspect
import plotly.subplots as sp

from ice_data_py import cds, hist3
from datetime import datetime

from scipy.constants import year
from datetime import datetime, timedelta
import calendar
import plotting.plotfig as plot
import functions as funct
import plotly.graph_objects as go

#######################################################################################################################
# Data Import #
#######################################################################################################################

# Define start and end time
t1 = datetime(2021, 6, 1, 0)
t2 = datetime(2025, 12, 31, 23)

print("=" * 70)
print("Loading hourly generation data from CDS")
print("Time Period:", t1, " to ", t2)
print("=" * 70)

## Load in hourly zones with zone-specific temperatures
print("\nLoading demand data...")
ct_hourly = cds.fetch_all([cds.QuerySpec(ms="NEISO", mn="load_zone._z_connecticut", mp="realtime_hourly_demand")], t1, t2)
wcmass_hourly = cds.fetch_all([cds.QuerySpec(ms="NEISO", mn="load_zone._z_wcmass", mp="realtime_hourly_demand")], t1, t2)
nh_hourly = cds.fetch_all([cds.QuerySpec(ms="NEISO", mn="load_zone._z_newhampshire", mp="realtime_hourly_demand")], t1, t2)

print("Loading zone-specific temperature data...")
ct_temp_hourly = cds.fetch_all([cds.QuerySpec(ms="NOAA-Forecast", mn="CT-Groton", mp="temperature[degF]")], t1, t2)
wcmass_temp_hourly = cds.fetch_all([cds.QuerySpec(ms="NOAA-Forecast", mn="MA-Worcester", mp="temperature[degF]")], t1, t2)
nh_temp_hourly = cds.fetch_all([cds.QuerySpec(ms="dev4-TWC-Forecasts", mn="VT.Georgia.coordinates", mp="temperature_F")], t1, t2)

# Convert to Polars DataFrames
ct_df = pl.DataFrame(ct_hourly)
wcmass_df = pl.DataFrame(wcmass_hourly)
nh_df = pl.DataFrame(nh_hourly)

ct_temp_df = pl.DataFrame(ct_temp_hourly)
wcmass_temp_df = pl.DataFrame(wcmass_temp_hourly)
nh_temp_df = pl.DataFrame(nh_temp_hourly)

print(f"CT: {len(ct_hourly)} demand records, {len(ct_temp_hourly)} temp records")
print(f"WMass: {len(wcmass_hourly)} demand records, {len(wcmass_temp_hourly)} temp records")
print(f"NH: {len(nh_hourly)} demand records, {len(nh_temp_hourly)} temp records")


#######################################################################################################################
# Process Zone Data #
#######################################################################################################################
print("\nProcessing zone data...")
ct_processed = funct.process_zone_data(ct_df, ct_temp_df, "Connecticut")
wcmass_processed = funct.process_zone_data(wcmass_df, wcmass_temp_df, "Western Mass")
nh_processed = funct.process_zone_data(nh_df, nh_temp_df, "New Hampshire")

print(f"Connecticut: {ct_processed.shape[0]} records")
print(f"Western Mass: {wcmass_processed.shape[0]} records")
print(f"New Hampshire: {nh_processed.shape[0]} records")

# Combine all zones
df = pl.concat([ct_processed, wcmass_processed, nh_processed])

# Extract temporal features
df = df.with_columns(
    pl.col("timestamp").dt.strftime("%Y-%m").alias("year_month"),
    pl.col("timestamp").dt.hour().alias("hour"),
    pl.col("timestamp").dt.year().alias("year"),
    pl.col("timestamp").dt.month().alias("month"),
    pl.col("timestamp").dt.day().alias("day"),
    pl.col("timestamp").dt.weekday().alias("dayofweek"),
)

# Create combined total zone
df_combined = df.group_by("timestamp", "year_month", "hour", "year", "month", "day", "dayofweek").agg(
    pl.col("value").sum().alias("value"),
    pl.col("temperature").mean().alias("temperature")  # Average temp across zones
).with_columns(
    pl.lit("Total (NH+CT+WMass)").alias("zone")
)

# Reorder columns to match df
df_combined = df_combined.select(df.columns)

# Add combined zone
df = pl.concat([df, df_combined])

# Get unique year-months sorted
year_months = sorted(df.select("year_month").unique().to_series().to_list())

print(f"\nFound {len(year_months)} months of data")
print("Months:")
for ym in year_months:
    month_obj = datetime.strptime(ym, "%Y-%m")
    pretty_month = month_obj.strftime("%B %Y")
    print(f"  - {pretty_month}")

print("\n" + "=" * 70)

df_combined = df.group_by("timestamp", "year_month", "hour", "year", "month", "day", "dayofweek").agg(
    pl.col("value").sum().alias("value"),
    pl.col("temperature").mean().alias("temperature")  # Average temp across zones
).with_columns(
    pl.lit("Total (NH+CT+WMass)").alias("zone")
)

# Reorder columns to match df
df_combined = df_combined.select(df.columns)



#######################################################################################################################
# Create Figure #
#######################################################################################################################

# Usage - with combined zone
zones = ["Connecticut", "Western Mass", "New Hampshire", "Total (NH+CT+WMass)"]

percentile_months = funct.calculate_percentiles_months(df, year_months)

percentile_years = funct.calculate_percentiles_years(df, year_months)

aggregate_percentiles = funct.aggregate_percentiles_by_month(percentile_years)

monthly_stats = funct.get_monthly_avg_of_daily_extremes(df, year_months)

load_file = False

if load_file:
    #######################################################################################################################
    # Load Existing Predictions #
    #######################################################################################################################

    print("\n" + "=" * 70)
    print("Loading Predictions from File")
    print("=" * 70)

    target_year = 2026
    predictions_file = f"predict/data/predictions_{target_year}.csv"

    try:
        predictions_combined = pl.read_csv(predictions_file)

        # Convert timestamp column to datetime
        predictions_combined = predictions_combined.with_columns(
            pl.col("timestamp").str.to_datetime().alias("timestamp")
        )

        print(f" Loaded predictions from: {predictions_file}")
        print(f"  Shape: {predictions_combined.shape}")
        print(f"  Columns: {predictions_combined.columns}")
        print(f"  Data types: {predictions_combined.dtypes}")
    except FileNotFoundError:
        print(f" Error: File not found - {predictions_file}")
        predictions_combined = None
    except Exception as e:
        print(f" Error loading file: {str(e)}")
        predictions_combined = None
else:
    #######################################################################################################################
    # Create Model #
    #######################################################################################################################

    print("\n" + "=" * 70)
    print("Engineering Features for ML Model")
    print("=" * 70)

    df_features = funct.engineer_features_with_weather(df)

    print(f"Before drop_nulls: {df_features.shape}")
    print(f"Zones before: {df_features.select('zone').unique().to_series().to_list()}")

    df_features = df_features.drop_nulls(subset=["value", "temperature", "hour_avg"])

    print(f"After drop_nulls: {df_features.shape}")
    print(f"Zones after: {df_features.select('zone').unique().to_series().to_list()}")

    # Check each zone
    for zone in zones:
        zone_count = df_features.filter(pl.col("zone") == zone).shape[0]
        print(f"  {zone}: {zone_count} records")

    print(f"\nColumns: {df_features.columns}")

    #######################################################################################################################
    # Train Models - Individual Zones Only #
    #######################################################################################################################

    print("\n" + "=" * 70)
    print("Training Models for Individual Zones Only")
    print("=" * 70)

    trained_models = {}
    zones_to_train = ["Connecticut", "Western Mass", "New Hampshire"]

    for zone in zones_to_train:
        df_zone = df_features.filter(pl.col("zone") == zone)
        trained_models[zone] = funct.train_zone_model_with_weather(df_zone, zone)

    #######################################################################################################################
    # Generate Predictions #
    #######################################################################################################################

    print("\n" + "=" * 70)
    print("Generating Predictions for Individual Zones")
    print("=" * 70)

    target_year = 2026
    predictions_all = []

    # Define zone-specific weather locations
    zone_weather_locations = {
        "Connecticut": "CT-Groton",
        "Western Mass": "MA-Worcester",
        "New Hampshire": "VT.Georgia.coordinates",
    }

    # Define weather data sources (some zones use different APIs)
    zone_weather_sources = {
        "Connecticut": ("NOAA-Forecast", "temperature[degF]"),
        "Western Mass": ("NOAA-Forecast", "temperature[degF]"),
        "New Hampshire": ("dev4-TWC-Forecasts", "temperature_F"),
    }

    for month in range(1, 7):
        month_name = datetime(target_year, month, 1).strftime('%B %Y')
        print(f"\n{month_name}:")

        # Define date range for this month
        t1_month = datetime(target_year, month, 1, 0)
        t2_month = datetime(target_year, month, 1) + timedelta(
            days=calendar.monthrange(target_year, month)[1]) - timedelta(seconds=1)
        t2_month = t2_month.replace(hour=23, minute=59, second=59)

        # Fetch zone-specific weather data for this month
        zone_weather_data = {}

        for zone in zones_to_train:
            try:
                location = zone_weather_locations[zone]
                ms, mp = zone_weather_sources[zone]

                weather_raw = cds.fetch_all(
                    [cds.QuerySpec(ms=ms, mn=location, mp=mp)],
                    t1_month,
                    t2_month
                )

                # Convert to Polars DataFrame
                weather_df = pl.DataFrame(weather_raw)

                # Standardize column names
                weather_df = weather_df.rename({weather_df.columns[1]: "temperature"})

                # Standardize timestamp
                temp_col = weather_df.columns[0]
                weather_df = weather_df.with_columns(
                    pl.col(temp_col).cast(pl.Datetime).alias("timestamp")
                )

                if temp_col != "timestamp":
                    weather_df = weather_df.drop(temp_col)

                zone_weather_data[zone] = weather_df

                if weather_df.shape[0] > 0:
                    avg_temp = weather_df.select("temperature").mean().item()
                    min_temp = weather_df.select("temperature").min().item()
                    max_temp = weather_df.select("temperature").max().item()
                    print(
                        f"  {zone}: {weather_df.shape[0]} records | Avg: {avg_temp:.1f}°F | Min: {min_temp:.1f}°F | Max: {max_temp:.1f}°F")
                else:
                    print(f"  {zone}: No weather data available")
                    zone_weather_data[zone] = None

            except Exception as e:
                print(f"  {zone}: Error fetching weather - {str(e)}")
                zone_weather_data[zone] = None

        # Generate predictions for individual zones only
        for zone in zones_to_train:
            try:
                pred_df = funct.predict_month_with_dynamic_adjustment(
                    zone=zone,
                    month=month,
                    year_to_predict=target_year,
                    df_input=df_features,
                    model_dict=trained_models[zone],
                    current_weather_data=zone_weather_data[zone]
                )

                if pred_df is not None and pred_df.shape[0] > 0:
                    predictions_all.append(pred_df)
                    print(f"   {zone}: {pred_df.shape[0]} predictions")
                else:
                    print(f"  {zone}: No predictions generated")
            except Exception as e:
                print(f"  {zone}: {str(e)}")
                continue

    #######################################################################################################################
    # Save Individual Zone Predictions #
    #######################################################################################################################

    predictions_combined = funct.save_predictions(
        predictions_list=predictions_all,
        target_year=target_year,
        output_dir="predict/data/"
    )

    #######################################################################################################################
    # Calculate Total Zone as Sum of Individual Zones #
    #######################################################################################################################

    if predictions_combined is not None and predictions_combined.shape[0] > 0:
        print("\n" + "=" * 70)
        print("Calculating Total Zone Predictions")
        print("=" * 70)

        total_predictions = []

        for month in range(1, 7):
            days_in_month = calendar.monthrange(target_year, month)[1]

            for day in range(1, days_in_month + 1):
                for hour in range(24):
                    try:
                        pred_date = datetime(target_year, month, day)

                        # Get predictions for each individual zone
                        ct_pred = predictions_combined.filter(
                            (pl.col("zone") == "Connecticut") &
                            (pl.col("timestamp").dt.date() == pred_date.date()) &
                            (pl.col("timestamp").dt.hour() == hour)
                        )

                        wm_pred = predictions_combined.filter(
                            (pl.col("zone") == "Western Mass") &
                            (pl.col("timestamp").dt.date() == pred_date.date()) &
                            (pl.col("timestamp").dt.hour() == hour)
                        )

                        nh_pred = predictions_combined.filter(
                            (pl.col("zone") == "New Hampshire") &
                            (pl.col("timestamp").dt.date() == pred_date.date()) &
                            (pl.col("timestamp").dt.hour() == hour)
                        )

                        # Sum if all three zones have predictions
                        if ct_pred.shape[0] > 0 and wm_pred.shape[0] > 0 and nh_pred.shape[0] > 0:
                            ct_mw = ct_pred.select("predicted_mw").item()
                            wm_mw = wm_pred.select("predicted_mw").item()
                            nh_mw = nh_pred.select("predicted_mw").item()
                            total_mw = ct_mw + wm_mw + nh_mw

                            # Sum percentiles and historical bounds
                            p10 = (ct_pred.select("p10").item() + wm_pred.select("p10").item() + nh_pred.select(
                                "p10").item())
                            p25 = (ct_pred.select("p25").item() + wm_pred.select("p25").item() + nh_pred.select(
                                "p25").item())
                            p75 = (ct_pred.select("p75").item() + wm_pred.select("p75").item() + nh_pred.select(
                                "p75").item())
                            p90 = (ct_pred.select("p90").item() + wm_pred.select("p90").item() + nh_pred.select(
                                "p90").item())
                            hist_avg = (ct_pred.select("historical_avg").item() + wm_pred.select(
                                "historical_avg").item() + nh_pred.select("historical_avg").item())
                            hist_min = (ct_pred.select("historical_min").item() + wm_pred.select(
                                "historical_min").item() + nh_pred.select("historical_min").item())
                            hist_max = (ct_pred.select("historical_max").item() + wm_pred.select(
                                "historical_max").item() + nh_pred.select("historical_max").item())

                            total_predictions.append({
                                'timestamp': pred_date.replace(hour=hour),
                                'zone': 'Total (NH+CT+WMass)',
                                'hour': hour,
                                'day': day,
                                'predicted_mw': float(total_mw),
                                'p10': float(p10),
                                'p25': float(p25),
                                'p75': float(p75),
                                'p90': float(p90),
                                'historical_min': float(hist_min),
                                'historical_max': float(hist_max),
                                'historical_avg': float(hist_avg),
                                'temperature_used': 0.0,
                                'temp_source': 'calculated',
                                'temp_adjustment_factor': 1.0,
                            })
                    except Exception as e:
                        continue

        # Add Total zone predictions to combined dataframe
        if total_predictions:
            total_df = pl.DataFrame(total_predictions)
            predictions_combined = pl.concat([predictions_combined, total_df])
            print(f"✓ Total zone calculated: {total_df.shape[0]} predictions")

#######################################################################################################################
# Create Visualizations #
#######################################################################################################################

if predictions_combined is not None and predictions_combined.shape[0] > 0:
    print("\n" + "=" * 70)
    print("Creating Visualizations")
    print("=" * 70)

    # Configuration: Change these to select which months to visualize
    VISUALIZATION_MONTHS = [1, 2, 3, 4, 5, 6]

    zones_for_viz = ["Connecticut", "Western Mass", "New Hampshire", "Total (NH+CT+WMass)"]

    for month in VISUALIZATION_MONTHS:
        plot.create_all_visualizations(
            predictions_df=predictions_combined,
            month=month,
            year=target_year,
            zones=zones_for_viz,
            output_dir="predict/figs/"
        )
else:
    print("\nERROR: Cannot create visualizations - no predictions available")
