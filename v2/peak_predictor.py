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

        # IMPORTANT: Fetch from beginning of month to today+7
        # This ensures max_peak includes all data from the month so far
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
        Find maximum peak and calculate predicted peak strength based on both actual and forecast

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

        today = at_time.date()
        today_midnight = datetime(today.year, today.month, today.day, 0, 0, 0)
        next_hour = at_time + timedelta(hours=1)

        # Convert to Polars datetime with timezone for comparison
        tz = pytz.timezone(TIMEZONE)
        today_midnight_tz = tz.localize(today_midnight)
        next_hour_tz = tz.localize(next_hour)

        # Get max actual load so far (up to today midnight)
        # This includes ALL data from beginning of month up to today
        df_up_to_today = df.filter(pl.col('datetime') <= pl.lit(today_midnight_tz))
        max_actual_load = df_up_to_today['load_mw'].max() if df_up_to_today.height > 0 else 0

        # Get init peak (if column exists)
        init_peak_mw = 0
        if 'init_peak_mw' in df.columns:
            init_peak_mw = df_up_to_today['init_peak_mw'].max() if df_up_to_today.height > 0 else 0

        # Get max forecast in entire dataset (to know what's predicted)
        max_forecast = df['forecast_3day_mw'].max() if df.height > 0 else 0

        # The target peak is the maximum of: actual load so far, init peak, or max forecast
        peak_values = [v for v in [max_actual_load, init_peak_mw, max_forecast] if v is not None and v > 0]

        if not peak_values:
            print("Warning: No peak values found")
            return pl.DataFrame()

        max_peak = max(peak_values)

        print(f"Max Peak: {max_peak} (actual: {max_actual_load}, init: {init_peak_mw}, forecast: {max_forecast})")

        # Filter to future hours
        df = df.filter(pl.col('datetime') >= pl.lit(next_hour_tz))

        if df.is_empty():
            return df

        # Calculate distance from peak for BOTH actual load and forecast
        df = df.with_columns([
            (pl.lit(max_peak) - pl.col('load_mw')).alias('actual_peak_diff_mw'),
            (pl.lit(max_peak) - pl.col('forecast_3day_mw')).alias('forecast_peak_diff_mw'),
        ])

        # Calculate peak strength for actual load
        df = df.with_columns([
            pl.when(pl.col('actual_peak_diff_mw') <= 0)
            .then(1.0)  # Actual load >= max_peak
            .when(pl.col('actual_peak_diff_mw') >= self.threshold_mw)
            .then(0.0)  # Actual load is threshold_mw below max_peak
            .otherwise(1.0 - (pl.col('actual_peak_diff_mw') / self.threshold_mw))
            .alias('actual_peak_strength')
        ])

        # Calculate peak strength for forecast
        df = df.with_columns([
            pl.when(pl.col('forecast_peak_diff_mw') <= 0)
            .then(1.0)  # Forecast >= max_peak
            .when(pl.col('forecast_peak_diff_mw') >= self.threshold_mw)
            .then(0.0)  # Forecast is threshold_mw below max_peak
            .otherwise(1.0 - (pl.col('forecast_peak_diff_mw') / self.threshold_mw))
            .alias('forecast_peak_strength')
        ])

        # Take the MAXIMUM of actual and forecast strength
        # This way, if either one is close to peak, it shows as a peak signal
        df = df.with_columns([
            pl.max_horizontal('actual_peak_strength', 'forecast_peak_strength')
            .alias('predicted_peak_strength')
        ])

        # Actionable strength
        df = df.with_columns([
            pl.when(pl.col('predicted_peak_strength') > 0)
            .then(pl.col('predicted_peak_strength'))
            .otherwise(0.0)
            .alias('predicted_peak_strength_actionable')
        ])

        return df
