# peak_predictor.py
"""
Peak prediction - combines load-based and temperature-weighted methods
"""

import polars as pl
from datetime import datetime, timedelta
import pytz
from helper import (
    fetch_df, add_datetime, add_weekday, add_upto_end_of_month, TIMEZONE
)


class PeakPredictorV3:
    """Predicts peak electrical usage with combined methods"""

    def __init__(self, zone, threshold_mw, derate_7day_forecast):
        self.zone = zone
        self.threshold_mw = threshold_mw
        self.derate_7day_forecast = derate_7day_forecast
        self.baseline_temp = 65.0  # Baseline temperature in Fahrenheit

    def calculate_temp_weight(self, apparent_temp, historical_avg_temp):
        """
        Calculate weight based on deviation from baseline and historical average
        Returns a multiplier (1.0 = normal, >1.0 = higher peak expected)
        """
        if apparent_temp is None or historical_avg_temp is None:
            return 1.0

        # Deviation from baseline
        deviation_from_baseline = abs(apparent_temp - self.baseline_temp)

        # Deviation from historical average
        deviation_from_historical = abs(apparent_temp - historical_avg_temp)

        # Weight is based on how much hotter/colder than baseline
        # Exponential weighting: extreme temps get higher weight
        weight = 1.0 + (deviation_from_baseline / 10.0) ** 1.5 * 0.3

        # Additional boost if significantly different from historical
        if deviation_from_historical > 5:
            weight *= 1.1

        return weight

    def run(self, data_columns, t_now, cds):
        """Run peak prediction for a given date"""
        if t_now.tzinfo is not None:
            tz = pytz.timezone(TIMEZONE)
            t_now = t_now.astimezone(tz).replace(tzinfo=None)

        today = t_now.date()
        today_plus7 = today + timedelta(days=7)

        # Fetch from beginning of month to today+7
        t_from = datetime(today.year, today.month, 1, 0, 0, 0)
        t_to = datetime(today_plus7.year, today_plus7.month, today_plus7.day, 0, 0, 0)

        full_df = fetch_df(data_columns, t_from, t_to, cds)

        if full_df.is_empty():
            return pl.DataFrame()

        current_month = today.month

        if today.month == 12:
            next_month_date = datetime(today.year + 1, 1, 1)
            next_month = 1
        else:
            next_month_date = datetime(today.year, today.month + 1, 1)
            next_month = today.month + 1

        this_month_df = full_df.filter(pl.col('month') == current_month)
        next_month_df = full_df.filter(pl.col('month') == next_month)

        this_month_df = self.find_max(this_month_df, t_now)
        next_month_df = self.find_max(next_month_df, next_month_date)

        if next_month_df.height > 0:
            final_df = pl.concat([this_month_df, next_month_df])
        else:
            final_df = this_month_df

        return final_df

    def find_max(self, df, at_time):
        """Find maximum peak and calculate predicted peak strength with combined methods"""
        if df.is_empty():
            return df

        if 'forecast_3day_mw' not in df.columns:
            return pl.DataFrame()

        df = df.drop_nulls(subset=['forecast_3day_mw'])

        if df.is_empty():
            return df

        today = at_time.date()
        today_midnight = datetime(today.year, today.month, today.day, 0, 0, 0)
        next_hour = at_time + timedelta(hours=1)

        tz = pytz.timezone(TIMEZONE)
        today_midnight_tz = tz.localize(today_midnight)
        next_hour_tz = tz.localize(next_hour)
        end_dt_7day = add_upto_end_of_month(at_time, 7)
        end_dt_7day_tz = tz.localize(end_dt_7day)
        at_time_tz = tz.localize(at_time)

        # CALCULATE HISTORICAL AVG TEMP FROM FULL DATASET FIRST (day 1 to today)
        historical_avg_temp = self.baseline_temp
        if 'apparent_temperature' in df.columns:
            df_full_month = df.filter(pl.col('datetime') <= pl.lit(today_midnight_tz))
            if df_full_month.height > 0:
                historical_avg_temp = df_full_month['apparent_temperature'].mean()

        # Get max actual load UP TO TODAY (cumulative)
        df_up_to_today = df.filter(pl.col('datetime') <= pl.lit(today_midnight_tz))
        max_peak_so_far = df_up_to_today['load_mw'].max() if df_up_to_today.height > 0 else 0

        # Get max forecast in NEXT 7 DAYS
        df_7day = df.filter((pl.col('datetime') >= pl.lit(at_time_tz)) & (pl.col('datetime') < pl.lit(end_dt_7day_tz)))
        max_7day_predicted_peak = df_7day['forecast_3day_mw'].max() if df_7day.height > 0 else 0

        # Calculate max peak
        peak_values = [v for v in [max_peak_so_far, max_7day_predicted_peak] if v is not None and v > 0]

        if not peak_values:
            return pl.DataFrame()

        max_peak = max(peak_values)

        print(f"Max Peak: {max_peak}, Historical Avg Temp: {historical_avg_temp}")

        # Filter to future hours
        df = df.filter(pl.col('datetime') >= pl.lit(next_hour_tz))

        if df.is_empty():
            return df

        # Calculate temperature weight using historical average from full month data
        if 'apparent_temperature' in df.columns:
            df = df.with_columns([
                pl.col('apparent_temperature').map_elements(
                    lambda x: self.calculate_temp_weight(x, historical_avg_temp),
                    return_dtype=pl.Float64
                ).alias('temp_weight')
            ])
        else:
            df = df.with_columns(pl.lit(1.0).alias('temp_weight'))

        # METHOD 1: Distance from peak for BOTH actual load and forecast
        df = df.with_columns([
            (pl.lit(max_peak) - pl.col('load_mw')).alias('actual_peak_diff_mw'),
            (pl.lit(max_peak) - pl.col('forecast_3day_mw')).alias('forecast_peak_diff_mw'),
        ])

        # Calculate peak strength for actual load
        df = df.with_columns([
            pl.when(pl.col('actual_peak_diff_mw') <= 0)
                .then(1.0)
                .when(pl.col('actual_peak_diff_mw') >= self.threshold_mw)
                .then(0.0)
                .otherwise(1.0 - (pl.col('actual_peak_diff_mw') / self.threshold_mw))
                .alias('actual_peak_strength')
        ])

        # Calculate peak strength for forecast
        df = df.with_columns([
            pl.when(pl.col('forecast_peak_diff_mw') <= 0)
                .then(1.0)
                .when(pl.col('forecast_peak_diff_mw') >= self.threshold_mw)
                .then(0.0)
                .otherwise(1.0 - (pl.col('forecast_peak_diff_mw') / self.threshold_mw))
                .alias('forecast_peak_strength')
        ])

        # Take the maximum of actual and forecast strength (METHOD 1)
        df = df.with_columns([
            pl.max_horizontal('actual_peak_strength', 'forecast_peak_strength')
                .alias('method1_peak_strength')
        ])

        # METHOD 2: Temperature-weighted forecast strength
        df = df.with_columns([
            (pl.lit(max_peak) - pl.col('forecast_3day_mw')).alias('predicted_peak_diff_mw'),
        ])

        df = df.with_columns([
            pl.when(pl.col('predicted_peak_diff_mw') <= 0)
                .then(1.0)
                .otherwise(1.0 - (pl.col('predicted_peak_diff_mw') / self.threshold_mw))
                .alias('predicted_peak_strength')
        ])

        # Apply temperature weighting to peak strength (METHOD 2)
        df = df.with_columns([
            (pl.col('predicted_peak_strength') * pl.col('temp_weight')).alias('method2_peak_strength')
        ])

        # COMBINE BOTH METHODS: Average them for better accuracy
        # This reduces false positives by requiring agreement between methods
        df = df.with_columns([
            ((pl.col('method1_peak_strength') + pl.col('method2_peak_strength')) / 2.0)
                .alias('predicted_peak_strength')
        ])

        # Actionable strength - only show if significant
        df = df.with_columns([
            pl.when(pl.col('predicted_peak_strength') > 0)
                .then(pl.col('predicted_peak_strength'))
                .otherwise(0.0)
                .alias('predicted_peak_strength_actionable')
        ])

        return df
