import numpy as np
import polars as pl

def calculate_percentiles_months(df, year_months, zones=None, percentiles=None):
    """
    Calculate percentiles for load data by zone and month, return as DataFrame.

    Parameters:
    -----------
    df : pl.DataFrame
        Polars DataFrame with columns: timestamp, value, zone, year_month
    year_months : list
        Sorted list of year-month strings (e.g., ["2024-06", "2024-07"])
    zones : list, optional
        List of zone names. Default: ["Connecticut", "Western Mass", "New Hampshire", "Total (NH+CT+WMass)"]
    percentiles : list, optional
        List of percentiles to calculate. Default: [90, 95, 97.5, 99]

    Returns:
    --------
    results_df : pl.DataFrame
        DataFrame with columns: year_month, zone, percentile_90, percentile_95, percentile_97_5, percentile_99
    """

    if zones is None:
        zones = ["Connecticut", "Western Mass", "New Hampshire", "Total (NH+CT+WMass)"]

    if percentiles is None:
        percentiles = [90, 95, 97.5, 99]

    data = []

    # Extract unique months (01-12) from all year_months
    df_with_month = df.with_columns(
        pl.col("year_month").str.split("-").list.get(1).alias("month")
    )

    unique_months = sorted(df_with_month.select("month").unique().to_series().to_list())

    # For each month, calculate percentiles across all years
    for month in unique_months:
        month_data = df_with_month.filter(pl.col("month") == month)

        for zone in zones:
            zone_data = month_data.filter(pl.col("zone") == zone).select("value").to_series()

            # Remove NaN and infinite values
            zone_data = zone_data.drop_nulls()
            zone_data_np = zone_data.to_numpy()
            zone_data_np = zone_data_np[np.isfinite(zone_data_np)]

            if len(zone_data_np) > 0:
                row = {"month": month, "zone": zone}
                for percentile in percentiles:
                    col_name = f"p{percentile}".replace(".", "_")
                    row[col_name] = np.percentile(zone_data_np, percentile)
                data.append(row)

    results_df = pl.DataFrame(data)
    return results_df


def calculate_percentiles_years(df, year_months, zones=None, percentiles=None):
    """
    Calculate percentiles for load data by zone, year, and month, return as DataFrame.

    Parameters:
    -----------
    df : pl.DataFrame
        Polars DataFrame with columns: timestamp, value, zone, year_month
    year_months : list
        Sorted list of year-month strings (e.g., ["2024-06", "2024-07"])
    zones : list, optional
        List of zone names. Default: ["Connecticut", "Western Mass", "New Hampshire", "Total (NH+CT+WMass)"]
    percentiles : list, optional
        List of percentiles to calculate. Default: [90, 95, 97.5, 99]

    Returns:
    --------
    results_df : pl.DataFrame
        DataFrame with columns: year, month, zone, percentile_90, percentile_95, percentile_97_5, percentile_99
    """

    if zones is None:
        zones = ["Connecticut", "Western Mass", "New Hampshire", "Total (NH+CT+WMass)"]

    if percentiles is None:
        percentiles = [90, 95, 97.5, 99]

    data = []

    # Extract year and month from year_month
    df_with_year_month = df.with_columns(
        pl.col("year_month").str.split("-").list.get(0).alias("year"),
        pl.col("year_month").str.split("-").list.get(1).alias("month")
    )

    unique_months = sorted(df_with_year_month.select("month").unique().to_series().to_list())
    unique_years = sorted(df_with_year_month.select("year").unique().to_series().to_list())

    # For each year and month combination, calculate percentiles
    for year in unique_years:
        for month in unique_months:
            month_data = df_with_year_month.filter(
                (pl.col("year") == year) & (pl.col("month") == month)
            )

            for zone in zones:
                zone_data = month_data.filter(pl.col("zone") == zone).select("value").to_series()

                # Remove NaN and infinite values
                zone_data = zone_data.drop_nulls()
                zone_data_np = zone_data.to_numpy()
                zone_data_np = zone_data_np[np.isfinite(zone_data_np)]

                if len(zone_data_np) > 0:
                    row = {"year": year, "month": month, "zone": zone}
                    for percentile in percentiles:
                        col_name = f"p{percentile}".replace(".", "_")
                        row[col_name] = np.percentile(zone_data_np, percentile)
                    data.append(row)

    results_df = pl.DataFrame(data)
    return results_df


def aggregate_percentiles_by_month(results_df):
    """
    Calculate average and median percentile values across all years for each month, zone, and percentile.

    Parameters:
    -----------
    results_df : pl.DataFrame
        DataFrame with columns: year, month, zone, p90, p95, p97_5, p99

    Returns:
    --------
    aggregated_df : pl.DataFrame
        DataFrame with columns: month, zone, percentile, average, median
        One row per month-zone-percentile combination
    """

    # Get all percentile columns (p90, p95, p97_5, p99)
    percentile_cols = [col for col in results_df.columns if col.startswith("p")]

    data = []

    # Get unique months and zones
    months = sorted(results_df.select("month").unique().to_series().to_list())
    zones = results_df.select("zone").unique().to_series().to_list()

    for month in months:
        for zone in zones:
            # Filter for specific month and zone
            month_zone_data = results_df.filter(
                (pl.col("month") == month) & (pl.col("zone") == zone)
            )

            for percentile_col in percentile_cols:
                values = month_zone_data.select(percentile_col).to_series()

                # Calculate average and median
                avg = values.mean()
                median = values.median()

                # Convert column name back to percentile label (p90 -> 90, p97_5 -> 97.5)
                percentile_label = percentile_col[1:].replace("_", ".")

                data.append({
                    "month": month,
                    "zone": zone,
                    "percentile": percentile_label,
                    "average": avg,
                    "median": median
                })

    aggregated_df = pl.DataFrame(data)
    return aggregated_df


def get_monthly_avg_of_daily_extremes(df, zones=None):
    """
    Calculate average of daily max/min/avg values for each month across all years.

    Process:
    1. For each year-month, calculate the average of daily max/min/avg values
    2. Then average those yearly values across all years

    Parameters:
    -----------
    df : pl.DataFrame
        Polars DataFrame with columns: timestamp, value, zone, year_month
    zones : list, optional
        List of zone names

    Returns:
    --------
    results_df : pl.DataFrame
        DataFrame with columns: month, zone, avg_max, avg_min, avg_avg
    """

    if zones is None:
        zones = ["Connecticut", "Western Mass", "New Hampshire", "Total (NH+CT+WMass)"]

    # Get timestamp column name
    timestamp_col = df.columns[0]

    # Extract year, month, and date
    df_with_dates = df.with_columns(
        pl.col(timestamp_col).dt.strftime("%Y").alias("year"),
        pl.col(timestamp_col).dt.strftime("%m").alias("month"),
        pl.col(timestamp_col).dt.date().alias("date")
    )

    # Step 1: Get daily max/min/avg for each day
    daily_stats = df_with_dates.group_by(["date", "zone"]).agg(
        pl.col("value").max().alias("daily_max"),
        pl.col("value").min().alias("daily_min"),
        pl.col("value").mean().alias("daily_avg")
    )

    # Add year and month back to daily stats
    daily_stats = daily_stats.with_columns(
        pl.col("date").dt.strftime("%Y").alias("year"),
        pl.col("date").dt.strftime("%m").alias("month")
    )

    # Step 2: For each year-month-zone, average the daily values
    yearly_monthly_stats = daily_stats.group_by(["year", "month", "zone"]).agg(
        pl.col("daily_max").mean().alias("yearly_avg_max"),
        pl.col("daily_min").mean().alias("yearly_avg_min"),
        pl.col("daily_avg").mean().alias("yearly_avg_avg")
    )

    # Step 3: For each month-zone, average across all years
    results_df = yearly_monthly_stats.group_by(["month", "zone"]).agg(
        pl.col("yearly_avg_max").mean().alias("avg_max"),
        pl.col("yearly_avg_min").mean().alias("avg_min"),
        pl.col("yearly_avg_avg").mean().alias("avg_avg")
    ).sort(["month", "zone"])

    return results_df


