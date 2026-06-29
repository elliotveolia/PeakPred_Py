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

# Load in hourly zones (WO/SOG Generation)
ct_hourly = cds.fetch_all([cds.QuerySpec(ms="NEISO", mn="load_zone._z_connecticut", mp="realtime_hourly_demand")], t1,
                          t2)
wcmass_hourly = cds.fetch_all([cds.QuerySpec(ms="NEISO", mn="load_zone._z_wcmass", mp="realtime_hourly_demand")], t1,
                              t2)
nh_hourly = cds.fetch_all([cds.QuerySpec(ms="NEISO", mn="load_zone._z_newhampshire", mp="realtime_hourly_demand")], t1,
                          t2)

#Load temperature data
temp_hourly = cds.fetch_all([cds.QuerySpec(ms="NOAA-Forecast", mn="MA-Boston", mp="temperature[degF]")], t1,
                          t2)


# Convert to Polars DataFrames
ct_df = pl.DataFrame(ct_hourly)
wcmass_df = pl.DataFrame(wcmass_hourly)
nh_df = pl.DataFrame(nh_hourly)
temp_df = pl.DataFrame(temp_hourly)

# Standardize column names and add zone
ct_df = ct_df.rename({ct_df.columns[1]: "value"}).with_columns(pl.lit("Connecticut").alias("zone"))
wcmass_df = wcmass_df.rename({wcmass_df.columns[1]: "value"}).with_columns(pl.lit("Western Mass").alias("zone"))
nh_df = nh_df.rename({nh_df.columns[1]: "value"}).with_columns(pl.lit("New Hampshire").alias("zone"))

# Standardize temperature column
temp_df = temp_df.rename({temp_df.columns[1]: "temperature"})

# Combine all zones
df = pl.concat([ct_df, wcmass_df, nh_df])

# Ensure timestamp column is datetime type, assuming first column is timestamp
df = df.with_columns(pl.col(df.columns[0]).cast(pl.Datetime).alias("timestamp"))

# Standardize temperature timestamp
temp_col = temp_df.columns[0]
temp_df = temp_df.with_columns(pl.col(temp_col).cast(pl.Datetime).alias("timestamp"))

# Extract year and month for grouping
df = df.with_columns(
    pl.col("timestamp").dt.strftime("%Y-%m").alias("year_month"),
    pl.col("timestamp").dt.hour().alias("hour")
)

# Drop the "timestamp" column from df - we only need the original column
if "timestamp" in df.columns and df.columns[0] != "timestamp":
    df = df.drop("timestamp")


# Drop the original temperature timestamp column if different
if temp_col != "timestamp":
    temp_df = temp_df.drop(temp_col)

# Combine all zones
df = pl.concat([ct_df, wcmass_df, nh_df])

# Ensure timestamp column is datetime type
first_col = df.columns[0]
df = df.with_columns(pl.col(first_col).cast(pl.Datetime).alias("timestamp"))

# Drop the original demand timestamp column if different
if first_col != "timestamp":
    df = df.drop(first_col)

# Extract year and month for grouping
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
    pl.col("value").sum().alias("value")
).with_columns(
    pl.lit("Total (NH+CT+WMass)").alias("zone")
)

# Reorder columns to match df exactly
df_combined = df_combined.select(df.columns)

# Add combined zone to the dataframe
df = pl.concat([df, df_combined])

# Join temperature data with demand data
print("Joining temperature data with demand data...")
df = df.join(temp_df.select(["timestamp", "temperature"]), on="timestamp", how="left")

# Get unique year-months sorted
year_months = sorted(df.select("year_month").unique().to_series().to_list())

print(f"Found {len(year_months)} months of data")
print("Months:")
for ym in year_months:
    month_obj = datetime.strptime(ym, "%Y-%m")
    pretty_month = month_obj.strftime("%B %Y")
    print(f"  - {pretty_month}")


#######################################################################################################################
# Create Figure #
#######################################################################################################################

# Usage - with combined zone
zones = ["Connecticut", "Western Mass", "New Hampshire", "Total (NH+CT+WMass)"]

percentile_months = funct.calculate_percentiles_months(df, year_months)

percentile_years = funct.calculate_percentiles_years(df, year_months)

aggregate_percentiles = funct.aggregate_percentiles_by_month(percentile_years)

monthly_stats = funct.get_monthly_avg_of_daily_extremes(df, year_months)



load_file = True

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
    df_features = df_features.drop_nulls()

    print(f"Features engineered. Shape: {df_features.shape}")
    print(f"Columns: {df_features.columns}")

    print("\n" + "=" * 70)
    print("Training Models for Each Zone")
    print("=" * 70)

    trained_models = {}

    for zone in zones:
        df_zone = df_features.filter(pl.col("zone") == zone)
        trained_models[zone] = funct.train_zone_model_with_weather(df_zone, zone)

    #######################################################################################################################
    # Generate Predictions #
    #######################################################################################################################

    print("\n" + "=" * 70)
    print("Generating Predictions")
    print("=" * 70)

    target_year = 2026
    predictions_all = []

    for month in range(1, 13):
        month_name = datetime(target_year, month, 1).strftime('%B %Y')
        print(f"\n{month_name}:")

        # Define date range for this month
        t1_month = datetime(target_year, month, 1, 0)
        t2_month = datetime(target_year, month, 1) + timedelta(
            days=calendar.monthrange(target_year, month)[1]) - timedelta(seconds=1)
        t2_month = t2_month.replace(hour=23, minute=59, second=59)

        # Fetch weather data for this month
        try:
            current_temps_raw = cds.fetch_all(
                [cds.QuerySpec(ms="NOAA-Forecast", mn="MA-Boston", mp="temperature[degF]")],
                t1_month,
                t2_month
            )

            # Convert to Polars DataFrame
            current_temps = pl.DataFrame(current_temps_raw)

            # Standardize column names
            current_temps = current_temps.rename({current_temps.columns[1]: "temperature"})

            # Standardize timestamp
            temp_col = current_temps.columns[0]
            current_temps = current_temps.with_columns(
                pl.col(temp_col).cast(pl.Datetime).alias("timestamp")
            )

            # Drop original timestamp column if different
            if temp_col != "timestamp":
                current_temps = current_temps.drop(temp_col)

            if current_temps.shape[0] > 0:
                avg_temp = current_temps.select("temperature").mean().item()
                min_temp = current_temps.select("temperature").min().item()
                max_temp = current_temps.select("temperature").max().item()
                print(
                    f"  Weather data: {current_temps.shape[0]} records | Avg: {avg_temp:.1f}°F | Min: {min_temp:.1f}°F | Max: {max_temp:.1f}°F")
            else:
                print(f"  No weather data available")
                current_temps = None

        except Exception as e:
            print(f"  Error fetching weather data: {str(e)}")
            current_temps = None

        for zone in zones:
            try:
                pred_df = funct.predict_month_with_dynamic_adjustment(
                    zone=zone,
                    month=month,
                    year_to_predict=target_year,
                    df_input=df_features,
                    model_dict=trained_models[zone],
                    current_weather_data=current_temps
                )

                if pred_df is not None and pred_df.shape[0] > 0:
                    predictions_all.append(pred_df)
                    print(f"  {zone}: {pred_df.shape[0]} predictions")
                else:
                    print(f"  {zone}: No predictions generated")
            except Exception as e:
                print(f"  {zone}: {str(e)}")
                continue

    #######################################################################################################################
    # Save Predictions #
    #######################################################################################################################

    predictions_combined = funct.save_predictions(
        predictions_list=predictions_all,
        target_year=target_year,
        output_dir="predict/data/"
    )


#######################################################################################################################
# Create Visualizations #
#######################################################################################################################

if predictions_combined is not None and predictions_combined.shape[0] > 0:
    # Configuration: Change these to select which months to visualize
    VISUALIZATION_MONTHS = [1, 2, 3, 4, 5, 6]  # January and February

    for month in VISUALIZATION_MONTHS:
        plot.create_all_visualizations(
            predictions_df=predictions_combined,
            month=month,
            year=target_year,
            zones=zones,
            output_dir="predict/figs/"
        )

    # Optional: Uncomment to create visualizations for all months
    # for month in range(1, 13):
    #     plot.create_all_visualizations(
    #         predictions_df=predictions_combined,
    #         month=month,
    #         year=target_year,
    #         zones=zones,
    #         output_dir="predict/figs/"
    #     )
else:
    print("\nERROR: Cannot create visualizations - no predictions available")