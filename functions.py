import numpy as np
import polars as pl

def calculate_percentiles_df(df, year_months, zones=None, percentiles=None):
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
