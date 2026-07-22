# helper.py
"""
Helper functions for data manipulation and analysis
"""

import polars as pl
import pytz
from datetime import datetime, timedelta

TIMEZONE = "America/New_York"


def fetch_df(data_columns, t1, t2, cds):
    """
    Fetch data from CDS and prepare it with datetime and derived columns

    Args:
        data_columns: dict of data sources to fetch
        t1: start time
        t2: end time
        cds: CDS module

    Returns:
        Polars DataFrame with processed data
    """
    # Fetch data from CDS
    dfs = []

    for col_name, spec in data_columns.items():
        try:
            print(f"\nFetching {col_name}...")

            # Build query spec
            query_spec = cds.QuerySpec(
                ms=spec['ms'],
                mn=spec['mn'],
                mp=spec['mp']
            )

            # Fetch data - returns data directly, not in a list
            data = cds.fetch_all([query_spec], t1, t2)

            if data is not None and len(data) > 0:
                # Ensure it's a Polars DataFrame
                if not isinstance(data, pl.DataFrame):
                    data = pl.DataFrame(data)

                df_temp = data.clone()

                # Debug: print column names and data
                print(f"=== {col_name} ===")
                print(f"Columns: {df_temp.columns}")
                print(f"Data shape: {df_temp.shape}")
                print(f"Data types:\n{df_temp.schema}")
                print(f"First few rows:\n{df_temp.head()}")
                print(f"Non-null counts:\n{df_temp.null_count()}")

                # Get column names
                cols = df_temp.columns

                if len(cols) < 2:
                    print(f"Warning: Expected at least 2 columns, got {len(cols)}")
                    continue

                # First column should be timestamp
                ts_col = cols[0]

                # Second column should be the value
                value_col = cols[1]
                df_temp = df_temp.select([ts_col, value_col])
                df_temp = df_temp.rename({ts_col: 'tstamp', value_col: col_name})

                # Drop rows where the value is null
                df_temp = df_temp.drop_nulls(subset=[col_name])

                if df_temp.height > 0:
                    dfs.append(df_temp)
                    print(f"✓ Successfully added {col_name} with {df_temp.height} rows")
                else:
                    print(f"✗ Warning: All values are null for {col_name}")

        except Exception as e:
            print(f"✗ Warning: Could not fetch {col_name}: {e}")
            import traceback
            traceback.print_exc()

    if not dfs:
        print("✗ Warning: No data fetched from CDS")
        return pl.DataFrame()

    # Join all dataframes on tstamp using coalesce to avoid duplicate columns
    df = dfs[0]
    for temp_df in dfs[1:]:
        df = df.join(temp_df, on='tstamp', how='outer', coalesce=True)

    # Sort by tstamp
    df = df.sort('tstamp')

    # Add datetime column
    df = add_datetime(df)

    # Add weekday
    df = add_weekday(df)

    # Add month and year
    df = df.with_columns([
        pl.col('datetime').dt.month().alias('month'),
        pl.col('datetime').dt.year().alias('year'),
        pl.col('datetime').dt.date().alias('he_date')
    ])

    print(f"\n✓ Final dataframe shape: {df.shape}")
    print(f"Columns: {df.columns}")

    return df


def add_datetime(df):
    """
    Convert unix timestamp to datetime in America/New_York timezone

    Args:
        df: Polars DataFrame with 'tstamp' column

    Returns:
        Polars DataFrame with added 'datetime' column
    """
    tz = TIMEZONE

    try:
        # Check if tstamp is already datetime
        if df['tstamp'].dtype in [pl.Datetime, pl.Date]:
            df = df.with_columns(pl.col('tstamp').alias('datetime'))
        else:
            # Assume unix timestamp in seconds, convert to datetime
            df = df.with_columns(
                pl.col('tstamp').cast(pl.Int64).cast(pl.Datetime('us')).dt.replace_time_zone(
                    'UTC').dt.convert_time_zone(tz).alias('datetime')
            )
    except Exception as e:
        print(f"Error converting datetime: {e}")
        print(f"tstamp dtype: {df['tstamp'].dtype}")
        print(f"Sample tstamp values: {df['tstamp'].head()}")

    return df


def add_weekday(df):
    """
    Add weekday column (0=Monday, 6=Sunday)

    Args:
        df: Polars DataFrame with 'datetime' column

    Returns:
        Polars DataFrame with added 'weekday' column
    """
    if 'datetime' in df.columns:
        df = df.with_columns(pl.col('datetime').dt.weekday().alias('weekday'))
    return df


def subtract_to_beg_of_month(at_time, days):
    """
    Subtract days from a datetime, but don't go before the beginning of the month

    Args:
        at_time: datetime object
        days: number of days to subtract

    Returns:
        datetime object
    """
    start_time = at_time - timedelta(days=days)

    if start_time.month == at_time.month:
        return start_time
    else:
        return datetime(at_time.year, at_time.month, 1, 0, 0, 0)


def add_upto_end_of_month(at_time, days):
    """
    Add days to a datetime, but don't go past the end of the month

    Args:
        at_time: datetime object
        days: number of days to add

    Returns:
        datetime object (midnight of the date)
    """
    start_time = datetime(at_time.year, at_time.month, at_time.day, 0, 0, 0)
    end_time = start_time + timedelta(days=days)

    if end_time.month == start_time.month:
        return end_time
    else:
        return datetime(end_time.year, end_time.month, 1, 0, 0, 0)


def features_to_numpy(df, features):
    """
    Convert selected features to numpy array

    Args:
        df: Polars DataFrame
        features: list of column names

    Returns:
        numpy array
    """
    return df.select(features).to_numpy()


def col_to_numpy(df, col_name):
    """
    Convert single column to numpy array

    Args:
        df: Polars DataFrame
        col_name: column name

    Returns:
        numpy array
    """
    return df[col_name].to_numpy()


def safe_series_first(series):
    """
    Safely get first value from a series

    Args:
        series: Polars Series

    Returns:
        First value or None
    """
    if isinstance(series, pl.Series) and len(series) > 0:
        return series[0]
    return None


def filter_peaks(df):
    """
    Filter dataframe to show only peak predictions

    Args:
        df: Polars DataFrame

    Returns:
        Filtered Polars DataFrame with peak data
    """
    if 'predicted_peak_strength' not in df.columns:
        return pl.DataFrame()

    df_filtered = df.filter(pl.col('predicted_peak_strength') > 0)

    cols_to_return = []
    for col in ['datetime', 'forecast_3day_mw', 'predicted_peak_strength']:
        if col in df_filtered.columns:
            cols_to_return.append(col)

    return df_filtered.select(cols_to_return) if cols_to_return else df_filtered
