#######################################################################################################################
# Imports #
#######################################################################################################################
import os
import sys

import polars as pl
import calendar
import plotting.plotfig as plot
import functions as funct

from ice_data_py import cds
from datetime import datetime, date
from datetime import datetime, timedelta

#######################################################################################################################
# Data Import #
#######################################################################################################################

# Define start and end time
t1 = datetime(2025, 6, 1, 0)
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

print("Loading zone-specific dew point data...")
ct_dewpoint_hourly = cds.fetch_all([cds.QuerySpec(ms="dev4-TWC-Forecasts", mn="MA.Northampton.coordinates", mp="dewPoint_F")], t1, t2)
wcmass_dewpoint_hourly = cds.fetch_all([cds.QuerySpec(ms="dev4-TWC-Forecasts", mn="MA.Worcester.coordinates", mp="dewPoint_F")], t1, t2)
nh_dewpoint_hourly = cds.fetch_all([cds.QuerySpec(ms="dev4-TWC-Forecasts", mn="VT.Georgia.coordinates", mp="dewPoint_F")], t1, t2)

# Convert to Polars DataFrames
ct_df = pl.DataFrame(ct_hourly)
wcmass_df = pl.DataFrame(wcmass_hourly)
nh_df = pl.DataFrame(nh_hourly)

ct_temp_df = pl.DataFrame(ct_temp_hourly)
wcmass_temp_df = pl.DataFrame(wcmass_temp_hourly)
nh_temp_df = pl.DataFrame(nh_temp_hourly)

ct_dewpoint_df = pl.DataFrame(ct_dewpoint_hourly)
wcmass_dewpoint_df = pl.DataFrame(wcmass_dewpoint_hourly)
nh_dewpoint_df = pl.DataFrame(nh_dewpoint_hourly)

print(f"CT: {len(ct_hourly)} demand records, {len(ct_temp_hourly)} temp records, {len(ct_dewpoint_hourly)} dewpoint records")
print(f"WMass: {len(wcmass_hourly)} demand records, {len(wcmass_temp_hourly)} temp records, {len(wcmass_dewpoint_hourly)} dewpoint records")
print(f"NH: {len(nh_hourly)} demand records, {len(nh_temp_hourly)} temp records, {len(nh_dewpoint_hourly)} dewpoint records")


#######################################################################################################################
# Process Zone Data #
#######################################################################################################################
print("\nProcessing zone data...")
ct_processed = funct.process_zone_data_with_dewpoint(ct_df, ct_temp_df, ct_dewpoint_df, "Connecticut")
wcmass_processed = funct.process_zone_data_with_dewpoint(wcmass_df, wcmass_temp_df, wcmass_dewpoint_df, "Western Mass")
nh_processed = funct.process_zone_data_with_dewpoint(nh_df, nh_temp_df, nh_dewpoint_df, "New Hampshire")

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
    pl.col("temperature").mean().alias("temperature"),
    pl.col("dewpoint").mean().alias("dewpoint"),
    pl.col("feels_like").mean().alias("feels_like")
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

#######################################################################################################################
# Create Figure #
#######################################################################################################################

# Usage - with combined zone
zones = ["Connecticut", "Western Mass", "New Hampshire", "Total (NH+CT+WMass)"]

percentile_months = funct.calculate_percentiles_months(df, year_months)

percentile_years = funct.calculate_percentiles_years(df, year_months)

aggregate_percentiles = funct.aggregate_percentiles_by_month(percentile_years)

monthly_stats = funct.get_monthly_avg_of_daily_extremes(df, year_months)

## main.py
"""
Main script to run peak prediction analysis
"""

import polars as pl
from datetime import datetime, timedelta
import os
from config import (
    DATA_SOURCES, PEAK_PREDICT_PARAMS, INIT_PEAK_FINDER_PARAMS,
    TIMEZONE, OUTPUT_DIR, OUTPUT_CSV_PEAK_PREDICT, OUTPUT_CSV_INIT_PEAK
)
from init_peak_finder import InitPeakFinder
from peak_predictor import PeakPredictor
from helper import fetch_df
from ice_data_py import cds


class PeakAnalysisRunner:
    """Main runner for peak analysis"""

    def __init__(self):
        """
        Initialize runner
        """
        self.cds = cds
        self.init_peak_finder = InitPeakFinder(
            alpha=INIT_PEAK_FINDER_PARAMS['default_alpha'],
            lookback_years=INIT_PEAK_FINDER_PARAMS['lookback_years']
        )
        self.peak_predictor = PeakPredictor(
            zone=PEAK_PREDICT_PARAMS['zone'],
            threshold_mw=PEAK_PREDICT_PARAMS['threshold_mw'],
            derate_7day_forecast=PEAK_PREDICT_PARAMS['derate_7day_forecast']
        )

        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def run_init_peak_finder(self, t1, t2):
        """
        Run initial peak finder for a date range

        Args:
            t1: start datetime
            t2: end datetime

        Returns:
            Polars DataFrame with results
        """
        print(f"Running InitPeakFinder from {t1} to {t2}")

        results = []
        current_date = t1.date() if hasattr(t1, 'date') else t1
        end_date = t2.date() if hasattr(t2, 'date') else t2

        while current_date <= end_date:
            current_datetime = datetime(
                current_date.year, current_date.month, current_date.day, 0, 0, 0
            )

            print(f"Processing date: {current_date}")

            result = self.init_peak_finder.run(DATA_SOURCES, current_datetime, self.cds)

            results.append({
                'date': current_date,
                'timestamp': result['timestamp'],
                'month': result['month'],
                'n95': result['n95'],
                'current_year_max': result['current_year_max'],
                'historical_min': result['historical_min'],
                'init_peak': result['init_peak']
            })

            current_date += timedelta(days=1)

        results_df = pl.DataFrame(results)

        # Save to CSV
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_INIT_PEAK)
        results_df.write_csv(output_path)
        print(f"Saved init peak finder results to {output_path}")

        return results_df

    def run_peak_predictor(self, t1, t2):
        """
        Run peak predictor for a date range

        Args:
            t1: start datetime
            t2: end datetime

        Returns:
            Polars DataFrame with predictions
        """
        print(f"Running PeakPredictor from {t1} to {t2}")

        results = []
        current_date = t1.date() if hasattr(t1, 'date') else t1
        end_date = t2.date() if hasattr(t2, 'date') else t2

        while current_date <= end_date:
            current_datetime = datetime(
                current_date.year, current_date.month, current_date.day, 0, 0, 0
            )

            print(f"Processing date: {current_date}")

            df = self.peak_predictor.run(DATA_SOURCES, current_datetime, self.cds)

            if not df.is_empty():
                results.append(df)

            current_date += timedelta(days=1)

        if results:
            results_df = pl.concat(results)
        else:
            results_df = pl.DataFrame()

        # Save to CSV
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_PEAK_PREDICT)
        results_df.write_csv(output_path)
        print(f"Saved peak predictor results to {output_path}")

        return results_df

    def run_all(self, t1, t2):
        """
        Run all analyses

        Args:
            t1: start datetime
            t2: end datetime
        """
        print("=" * 80)
        print("PEAK ELECTRICAL USAGE ANALYSIS")
        print("=" * 80)

        init_peak_results = self.run_init_peak_finder(t1, t2)
        print("\nInitial Peak Finder Results:")
        print(init_peak_results)

        print("\n" + "=" * 80 + "\n")

        peak_predict_results = self.run_peak_predictor(t1, t2)
        print("\nPeak Predictor Results:")
        print(peak_predict_results)

        print("\n" + "=" * 80)
        print("Analysis complete!")
        print(f"Results saved to {OUTPUT_DIR}/")


# Example usage
if __name__ == "__main__":
    if __name__ == "__main__":
        # Initialize runner
        runner = PeakAnalysisRunner()

        # Define date range - using 2024 data
        t1 = datetime(2024, 1, 1)
        t2 = datetime(2024, 1, 31)

        # Run analysis
        runner.run_all(t1, t2)