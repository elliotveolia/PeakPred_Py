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

import plotting.plotfig as plot
import functions as funct
import plotly.graph_objects as go

#######################################################################################################################
# Data Import #
#######################################################################################################################

# Define start and end time
t1 = datetime(2021, 6, 1, 0)
t2 = datetime(2026, 6, 1, 23)

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

# Convert to Polars DataFrames
ct_df = pl.DataFrame(ct_hourly)
wcmass_df = pl.DataFrame(wcmass_hourly)
nh_df = pl.DataFrame(nh_hourly)

# Standardize column names and add zone
ct_df = ct_df.rename({ct_df.columns[1]: "value"}).with_columns(pl.lit("Connecticut").alias("zone"))
wcmass_df = wcmass_df.rename({wcmass_df.columns[1]: "value"}).with_columns(pl.lit("Western Mass").alias("zone"))
nh_df = nh_df.rename({nh_df.columns[1]: "value"}).with_columns(pl.lit("New Hampshire").alias("zone"))

# Combine all zones
df = pl.concat([ct_df, wcmass_df, nh_df])

# Ensure timestamp column is datetime type, assuming first column is timestamp
df = df.with_columns(pl.col(df.columns[0]).cast(pl.Datetime).alias("timestamp"))

# Extract year and month for grouping
df = df.with_columns(
    pl.col("timestamp").dt.strftime("%Y-%m").alias("year_month"),
    pl.col("timestamp").dt.hour().alias("hour")
)

# Drop the "timestamp" column from df - we only need the original column
if "timestamp" in df.columns and df.columns[0] != "timestamp":
    df = df.drop("timestamp")

# Create combined total zone
first_col = df.columns[0]
df_combined = df.group_by(first_col, "year_month", "hour").agg(
    pl.col("value").sum().alias("value")
).with_columns(
    pl.lit("Total (NH+CT+WMass)").alias("zone")
)

# Reorder columns to match df exactly
df_combined = df_combined.select(df.columns)



# Add combined zone to the dataframe
df = pl.concat([df, df_combined])

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

#fig = create_load_profile_figure(df, year_months, zones=zones)
#fig.show()

#fig = create_load_distribution_figure(df, year_months,)
#fig.show()

percentile_months = funct.calculate_percentiles_months(df, year_months)

#fig = create_load_profile_figure_percent(df, year_months, percentile_data=percentile_months)
#fig.show()

percentile_years = funct.calculate_percentiles_years(df, year_months)

aggregate_percentiles = funct.aggregate_percentiles_by_month(percentile_years)

#fig_avg = create_load_percentile_figure(
#    df,
#    year_months,
#    percentile_data=aggregate_percentiles,
#    output_file="load_profile_average.html",
#    percentile_type="average"
#)

#fig_avg = create_load_percentile_figure(
#    df,
#    year_months,
#    percentile_data=aggregate_percentiles,
#    output_file="load_profile_median.html",
#    percentile_type="median"
#)

monthly_stats = funct.get_monthly_avg_of_daily_extremes(df, year_months)

#fig = plot.create_load_profile_figure_minmax(
#    df=df,
#    year_months=year_months,
#    minmax_data=monthly_stats,
#    output_file="load_profile_minmax.html"
#)


#fig = plot.create_peak_calendar_from_percentiles(
#    df=df,
#    percentiles_df=percentile_years,
#    percentile_threshold=99,
#    output_file="peak_calendar_percentiles.html"
#)

#######################################################################################################################
# Create Model #
#######################################################################################################################

print("\n" + "=" * 70)
print("Engineering Features for ML Model")
print("=" * 70)

df_features = funct.engineer_features(df)
df_features = df_features.drop_nulls()

print(f"Features engineered. Shape: {df_features.shape}")
print(f"Columns: {df_features.columns}")

trained_models = {}

print("\n" + "=" * 70)
print("Training Models for Each Zone")
print("=" * 70)

for zone in zones:
    df_zone = df_features.filter(pl.col("zone") == zone)
    trained_models[zone] = funct.train_zone_model(df_zone, zone)

target_year = 2026
predictions_all = []

for month in range(1, 2):
    month_name = datetime(target_year, month, 1).strftime('%B %Y')
    print(f"\nPredicting {month_name}...")

    for zone in zones:
        try:
            pred_df = funct.predict_month(
                zone=zone,
                month=month,
                year_to_predict=target_year,
                df_input=df_features,
                model_dict=trained_models[zone],
                percentile_data=None
            )

            if pred_df is not None and pred_df.shape[0] > 0:
                predictions_all.append(pred_df)
                print(f"  {zone}: {pred_df.shape[0]} predictions")
            else:
                print(f"  {zone}: No predictions generated")
        except Exception as e:
            print(f"  {zone}: Error - {str(e)}")
            continue

# Combine all predictions
if len(predictions_all) > 0:
    predictions_combined = pl.concat(predictions_all)
    print(f"\nTotal predictions generated: {predictions_combined.shape[0]}")
    print(f"Columns: {predictions_combined.columns}")
    print(f"\nFirst 24 rows (first day):")
    print(predictions_combined.head(24))

    # Save predictions
    predictions_combined.write_csv(f"predict/data/predictions_{target_year}.csv")
    print(f"\nPredictions saved to predictions_{target_year}.csv")
else:
    print("ERROR: No predictions were generated!")
    predictions_combined = None

if predictions_combined is not None and predictions_combined.shape[0] > 0:
    print("\n" + "=" * 70)
    print("Creating Visualizations")
    print("=" * 70)

    # Create predictions for all zones in January
    for zone in zones:
        try:
            fig = plot.create_prediction_figure(predictions_combined, month=1, year=2026, zone=zone)
            if fig is not None:
                filename = f"prediction_jan_2026_{zone.replace(' ', '_').replace('(', '').replace(')', '').lower()}.html"
                fig.write_html("predict/figs/" + filename)
                print(f"Saved: {filename}")
        except Exception as e:
            print(f"Error creating figure for {zone}: {str(e)}")

    # Create a summary figure for all zones
    print("\nCreating summary figure...")
    try:
        fig_summary = go.Figure()

        for zone in zones:
            pred_zone = predictions_combined.filter(
                (pl.col("zone") == zone) &
                (pl.col("timestamp").dt.month() == 1) &
                (pl.col("timestamp").dt.year() == 2026)
            )

            if pred_zone.shape[0] > 0:
                timestamps = pred_zone.select("timestamp").to_series().to_list()
                predicted_mw = pred_zone.select("predicted_mw").to_series().to_list()

                fig_summary.add_trace(go.Scatter(
                    x=timestamps,
                    y=predicted_mw,
                    mode='lines',
                    name=zone,
                    line=dict(width=2)
                ))

        fig_summary.update_layout(
            title="Predicted MW Usage - All Zones - January 2026",
            xaxis_title="Date",
            yaxis_title="Megawatts",
            hovermode='x unified',
            template='plotly_white',
            height=600,
            width=1200
        )

        fig_summary.write_html("prediction_jan_2026_all_zones.html")
        print("Saved: prediction_jan_2026_all_zones.html")
    except Exception as e:
        print(f"Error creating summary figure: {str(e)}")

    print("\n" + "=" * 70)
    print("Prediction visualizations saved!")
    print("=" * 70)
else:
    print("ERROR: Cannot create visualizations - no predictions available")
