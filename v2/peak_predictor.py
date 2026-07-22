# peak_predictor.py
"""
Peak prediction - predicts peak electrical usage based on 95th percentile
"""

import polars as pl
from datetime import datetime, timedelta
import pytz
from helper import (
    fetch_df, add_datetime, add_weekday, add_upto_end_of_month, TIMEZONE
)


class PeakPredictor:
    """Predicts peak electrical usage"""

    def __init__(self, zone, threshold_mw, derate_7day_forecast):
        """
        Initialize peak predictor

        Args:
            zone: zone name (e.g., "Connecticut")
            threshold_mw: threshold in MW for peak calculation
            derate_7day_forecast: derate factor for 7-day forecast
        """
        self.zone = zone
        self.threshold_mw = threshold_mw
        self.derate_7day_forecast = derate_7day_forecast

    def run(self, data_columns, t_now, cds):
        """
        Run peak prediction for a given date

        Args:
            data_columns: dict of data sources
            t_now: datetime to predict for
            cds: CDS client instance

        Returns:
            Polars DataFrame with predictions
        """
        # Convert to naive datetime if needed
        if t_now.tzinfo is not None:
            tz = pytz.timezone(TIMEZONE)
            t_now = t_now.astimezone(tz).replace(tzinfo=None)

        today = t_now.date()
        today_plus7 = today + timedelta(days=7)

        # Calculate time range
        t_from = datetime(today.year, today.month, 1, 0, 0, 0)
        t_to = datetime(today_plus7.year, today_plus7.month, today_plus7.day, 0, 0, 0)

        # Fetch data
        full_df = fetch_df(data_columns, t_from, t_to, cds)

        if full_df.is_empty():
            return pl.DataFrame()

        current_month = today.month

        # Calculate next month
        if today.month == 12:
            next_month_date = datetime(today.year + 1, 1, 1)
            next_month = 1
        else:
            next_month_date = datetime(today.year, today.month + 1, 1)
            next_month = today.month + 1

        # Split by month
        this_month_df = full_df.filter(pl.col('month') == current_month)
        next_month_df = full_df.filter(pl.col('month') == next_month)

        # Find max for each month
        this_month_df = self.find_max(this_month_df, t_now)
        next_month_df = self.find_max(next_month_df, next_month_date)

        # Concatenate
        if next_month_df.height > 0:
            final_df = pl.concat([this_month_df, next_month_df])
        else:
            final_df = this_month_df

        return final_df

    def find_max(self, df, at_time):
        """
        Find maximum peak and calculate predicted peak strength

        Args:
            df: Polars DataFrame with load and forecast data
            at_time: datetime to calculate from

        Returns:
            Polars DataFrame with peak predictions
        """
        if df.is_empty():
            return df

        # Check if forecast_3day_mw column exists
        if 'forecast_3day_mw' not in df.columns:
            print(f"Warning: forecast_3day_mw column not found. Available columns: {df.columns}")
            return pl.DataFrame()

        # Remove rows with missing forecast data
        df = df.drop_nulls(subset=['forecast_3day_mw'])

        if df.is_empty():
            print("Warning: All forecast_3day_mw values are null")
            return df

        next_hour = at_time + timedelta(hours=1)
        today = at_time.date()
        today_midnight = datetime(today.year, today.month, today.day, 0, 0, 0)

        # Convert to Polars datetime with timezone for comparison
        tz = pytz.timezone(TIMEZONE)
        today_midnight_tz = tz.localize(today_midnight)
        next_hour_tz = tz.localize(next_hour)
        end_dt_7day = add_upto_end_of_month(at_time, 7)
        end_dt_7day_tz = tz.localize(end_dt_7day)
        at_time_tz = tz.localize(at_time)

        # Get max peak so far (up to today midnight)
        df_up_to_today = df.filter(pl.col('datetime') <= pl.lit(today_midnight_tz))
        max_peak_so_far = df_up_to_today['load_mw'].max() if df_up_to_today.height > 0 else None

        # Get init peak (if column exists)
        init_peak_mw = None
        if 'init_peak_mw' in df.columns:
            init_peak_mw = df_up_to_today['init_peak_mw'].max() if df_up_to_today.height > 0 else None

        # Get 7-day forecast max
        df_7day = df.filter((pl.col('datetime') >= pl.lit(at_time_tz)) & (pl.col('datetime') < pl.lit(end_dt_7day_tz)))
        max_7day_predicted_peak = df_7day['forecast_3day_mw'].max() if df_7day.height > 0 else None

        # Calculate max peak - filter out None values
        peak_values = [v for v in [max_peak_so_far, init_peak_mw, max_7day_predicted_peak] if v is not None]

        if not peak_values:
            print("Warning: No peak values found")
            return pl.DataFrame()

        max_peak = max(peak_values)

        print(f"Max Peak: {max_peak}")

        # Filter to future hours
        df = df.filter(pl.col('datetime') >= pl.lit(next_hour_tz))

        if df.is_empty():
            return df

        # Calculate predicted peak difference and strength
        df = df.with_columns([
            (pl.lit(max_peak) - pl.col('forecast_3day_mw')).alias('predicted_peak_diff_mw'),
        ])

        df = df.with_columns([
            pl.when(pl.col('predicted_peak_diff_mw') <= 0)
            .then(1.0)
            .otherwise(1.0 - (pl.col('predicted_peak_diff_mw') / self.threshold_mw))
            .alias('predicted_peak_strength')
        ])

        df = df.with_columns([
            pl.when(pl.col('predicted_peak_strength') > 0)
            .then(pl.col('predicted_peak_strength'))
            .otherwise(0.0)
            .alias('predicted_peak_strength_actionable')
        ])

        return df
