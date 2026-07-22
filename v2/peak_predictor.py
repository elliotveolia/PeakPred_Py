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
        self.zone = zone
        self.threshold_mw = threshold_mw
        self.derate_7day_forecast = derate_7day_forecast

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
        """Find maximum peak and calculate predicted peak strength"""
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

        # Max actual load UP TO TODAY (cumulative)
        df_up_to_today = df.filter(pl.col('datetime') <= pl.lit(today_midnight_tz))
        max_peak_so_far = df_up_to_today['load_mw'].max() if df_up_to_today.height > 0 else 0

        # Max forecast in NEXT 7 DAYS
        df_7day = df.filter((pl.col('datetime') >= pl.lit(at_time_tz)) & (pl.col('datetime') < pl.lit(end_dt_7day_tz)))
        max_7day_predicted_peak = df_7day['forecast_3day_mw'].max() if df_7day.height > 0 else 0

        # DON'T use init_peak_mw - it's an outlier
        # Just use the max of actual load so far and 7-day forecast
        peak_values = [v for v in [max_peak_so_far, max_7day_predicted_peak] if v is not None and v > 0]

        if not peak_values:
            return pl.DataFrame()

        max_peak = max(peak_values)

        print(f"Max Peak: {max_peak} (actual_so_far: {max_peak_so_far}, 7day_forecast: {max_7day_predicted_peak})")

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

        # Take the maximum of actual and forecast strength
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
