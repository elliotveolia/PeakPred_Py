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
print(monthly_stats)


fig = plot.create_load_profile_figure_minmax(
    df=df,
    year_months=year_months,
    minmax_data=monthly_stats,
    output_file="load_profile_minmax.html"
)

