# init_peak_finder.py
"""
Initial peak finder - calculates 95th percentile of historical data
"""

import polars as pl
from datetime import datetime, timedelta
import pytz
from helper import (
    fetch_df, add_datetime, add_weekday, safe_series_first,
    subtract_to_beg_of_month, TIMEZONE
)


class InitPeakFinder:
    """Finds initial peak based on 95th percentile of historical data"""

    def __init__(self, alpha=0.1, lookback_years=3):
        """
        Initialize peak finder

        Args:
            alpha: smoothing factor (0.0-1.0)
            lookback_years: number of years to look back for historical data
        """
        self.alpha = max(0.0, min(1.0, alpha))
        self.lookback_years = lookback_years

    def run(self, data_columns, t_now, cds):
        """
        Calculate initial peak for a given date

        Args:
            data_columns: dict of data sources
            t_now: datetime to calculate peak for
            cds: CDS client instance

        Returns:
            dict with peak analysis results
        """
        # Convert to naive datetime if needed
        if t_now.tzinfo is not None:
            tz = pytz.timezone(TIMEZONE)
            t_now = t_now.astimezone(tz).replace(tzinfo=None)

        # Calculate lookback period
        t_now_minus_3_year = datetime(
            t_now.year - self.lookback_years,
            t_now.month,
            1,
            0, 0, 0
        )

        # Fetch data
        df = fetch_df(data_columns, t_now_minus_3_year, t_now, cds)

        if df.is_empty():
            return self._empty_result(t_now)

        # Filter to current month
        monthly_df = df.filter(pl.col('month') == t_now.month)

        # Daily max for selected month over last 3 completed years
        historical_df = monthly_df.filter(pl.col('year') < t_now.year)

        if historical_df.height > 0:
            daily_max_df = historical_df.group_by('he_date').agg(
                pl.col('load_mw').max().alias('max_load')
            ).with_columns(
                pl.col('he_date').cast(pl.Date).dt.year().alias('year')
            )
        else:
            daily_max_df = pl.DataFrame()

        # Current year max
        current_year_df = monthly_df.filter(pl.col('year') == t_now.year)
        current_year_max = current_year_df['load_mw'].max() if current_year_df.height > 0 else None

        # 95th percentile of historical data
        n95 = None
        if daily_max_df.height > 0:
            n95 = daily_max_df['max_load'].quantile(0.95)

        # Max daily load for each year
        if daily_max_df.height > 0:
            year_max_df = daily_max_df.group_by('year').agg(
                pl.col('max_load').max().alias('year_max')
            )
        else:
            year_max_df = pl.DataFrame()

        year_max_values = year_max_df['year_max'].to_list() if year_max_df.height > 0 else []
        year_max_values = [x for x in year_max_values if x is not None]

        # Historical minimum (95th percentile or year max)
        candidates = [n95] + year_max_values
        candidates = [x for x in candidates if x is not None]
        historical_min = min(candidates) if candidates else None

        # Calculate initial peak
        init_peak = self._calculate_init_peak(current_year_max, historical_min)

        return {
            'month': t_now.month,
            'alpha': self.alpha,
            'n95': n95,
            'current_year_max': current_year_max,
            'year_max_df': year_max_df,
            'historical_min': historical_min,
            'init_peak': init_peak,
            'timestamp': t_now
        }

    def _calculate_init_peak(self, current_year_max, historical_min):
        """
        Calculate initial peak from current year max and historical minimum

        Args:
            current_year_max: max load in current year
            historical_min: minimum of 95th percentile and year maxes

        Returns:
            Initial peak value or None
        """
        values = []
        if current_year_max is not None:
            values.append(current_year_max)
        if historical_min is not None:
            values.append(historical_min)

        return max(values) if values else None

    def _empty_result(self, t_now):
        """Return empty result structure"""
        return {
            'month': t_now.month,
            'alpha': self.alpha,
            'n95': None,
            'current_year_max': None,
            'year_max_df': pl.DataFrame(),
            'historical_min': None,
            'init_peak': None,
            'timestamp': t_now
        }
