import polars as pl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from datetime import datetime


def create_load_profile_figure(df, year_months, zones=None, output_file="load_profile.html"):
    """
    Create an interactive load profile figure with month/year dropdown selector.

    Parameters:
    -----------
    df : pl.DataFrame
        Polars DataFrame with columns: timestamp, value, zone, year_month
    year_months : list
        Sorted list of year-month strings (e.g., ["2024-06", "2024-07"])
    zones : list, optional
        List of zone names. Default: ["Connecticut", "Western Mass", "New Hampshire"]
    output_file : str, optional
        Path to save HTML file. Default: "load_profile.html"

    Returns:
    --------
    fig : go.Figure
        Plotly figure object
    """

    if zones is None:
        zones = ["Connecticut", "Western Mass", "New Hampshire"]

    # Get the actual timestamp column name (first column)
    timestamp_col = df.columns[0]

    # Create figure
    fig = go.Figure()

    # Create traces for each year-month combination
    for year_month in year_months:
        month_data = df.filter(pl.col("year_month") == year_month)

        for zone in zones:
            zone_data = month_data.filter(pl.col("zone") == zone).sort(timestamp_col)

            if len(zone_data) > 0:  # Only add if data exists
                fig.add_trace(
                    go.Scatter(
                        x=zone_data.select(timestamp_col).to_series(),
                        y=zone_data.select("value").to_series(),
                        mode="lines",
                        name=zone,
                        visible=(year_month == year_months[0]),  # Only first month visible initially
                        hovertemplate=f"<b>{zone}</b><br>Time: %{{x}}<br>Load: %{{y:.2f}} MW<extra></extra>",
                        legendgroup=zone,
                        showlegend=(year_month == year_months[0])
                    )
                )

    # Create buttons for month selection
    buttons = []
    for i, year_month in enumerate(year_months):
        # Create visibility list
        visibility = []
        for j, ym in enumerate(year_months):
            for zone in zones:
                visibility.append(ym == year_month)

        # Convert to pretty format for display
        month_obj = datetime.strptime(year_month, "%Y-%m")
        pretty_label = month_obj.strftime("%B %Y")

        buttons.append(
            dict(
                label=pretty_label,
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"Load Profile - {pretty_label}"}
                ]
            )
        )

    # Update layout with dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ],
        title=f"Load Profile - {datetime.strptime(year_months[0], '%Y-%m').strftime('%B %Y')}",
        xaxis_title="Time",
        yaxis_title="Load (MW)",
        hovermode="x unified",
        height=600,
        template="plotly_white",
        legend=dict(
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top"
        )
    )

    # Save as HTML
    fig.write_html(output_file)

    return fig


def create_load_distribution_figure(df, year_months, zones=None, output_file="load_distribution.html"):
    """
    Create an interactive load distribution figure with month/year dropdown selector.
    Shows smooth KDE density curves for each zone in separate subplots with percentile lines and annotations.
    """

    if zones is None:
        zones = ["Connecticut", "Western Mass", "New Hampshire"]

    # Define percentiles and their colors
    percentiles = [90, 95, 97.5, 99]
    percentile_colors = {90: "green", 95: "blue", 97.5: "orange", 99: "red"}

    # Define colors for each zone
    zone_colors = {
        "Connecticut": "#1f77b4",
        "Western Mass": "#ff7f0e",
        "New Hampshire": "#2ca02c",
        "Total (NH+CT+WMass)": "#d62728"
    }

    # Create figure with subplots (one for each zone)
    fig = sp.make_subplots(
        rows=1, cols=len(zones),
        subplot_titles=zones,
        specs=[[{"secondary_y": False} for _ in zones]],
        shared_xaxes=False
    )

    # Store annotations for each month
    annotations_by_month = {ym: [] for ym in year_months}

    # Create traces for each year-month combination
    for year_month in year_months:
        month_data = df.filter(pl.col("year_month") == year_month)

        for col_idx, zone in enumerate(zones, 1):
            zone_data = month_data.filter(pl.col("zone") == zone).select("value").to_series()

            # Remove NaN and infinite values
            zone_data = zone_data.drop_nulls()
            zone_data_np = zone_data.to_numpy()
            zone_data_np = zone_data_np[np.isfinite(zone_data_np)]

            if len(zone_data_np) > 1:  # Need at least 2 points for KDE
                from scipy.stats import gaussian_kde

                # Calculate all percentile values from raw data
                percentile_values = {p: np.percentile(zone_data_np, p) for p in percentiles}
                max_percentile = max(percentile_values.values())

                # Create x_range that extends well beyond the max percentile
                x_min = zone_data_np.min()
                x_max = max_percentile * 1.15  # Extend 15% beyond max percentile
                x_range = np.linspace(x_min, x_max, 300)  # More points for accuracy

                kde = gaussian_kde(zone_data_np)
                density = kde(x_range)

                # Normalize density to 0-1 range for better readability
                density_normalized = density / density.max()
                max_density = 1.0

                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=density_normalized,
                        mode='lines',
                        name=zone,
                        visible=(year_month == year_months[0]),
                        hovertemplate=f"<b>{zone}</b><br>Load: %{{x:.2f}} MW<br>Density: %{{y:.4f}}<extra></extra>",
                        legendgroup=zone,
                        showlegend=(year_month == year_months[0]),
                        line=dict(width=2, color=zone_colors.get(zone, "#000000")),
                        fill='tozeroy',
                        opacity=0.6
                    ),
                    row=1, col=col_idx
                )

                # Add percentile lines with annotations
                for percentile in percentiles:
                    percentile_value = percentile_values[percentile]

                    # Find the density value at this percentile point on the normalized KDE curve
                    density_at_percentile = np.interp(percentile_value, x_range, density_normalized)

                    fig.add_trace(
                        go.Scatter(
                            x=[percentile_value, percentile_value],
                            y=[0, density_at_percentile],  # Use normalized density
                            mode='lines',
                            name=f"{percentile}th percentile",
                            visible=(year_month == year_months[0]),
                            hovertemplate=f"<b>{percentile}th percentile</b><br>Load: {percentile_value:.2f} MW<extra></extra>",
                            line=dict(color=percentile_colors[percentile], width=2, dash='dash'),
                            legendgroup=f"percentile_{percentile}",
                            showlegend=False,
                        ),
                        row=1, col=col_idx
                    )

                    # Store annotation for this month
                    annotations_by_month[year_month].append(
                        dict(
                            x=percentile_value,
                            y=density_at_percentile + 0.05,  # Position above the line
                            text=f"{percentile}%<br>{percentile_value:.0f}",
                            showarrow=False,
                            font=dict(size=9, color=percentile_colors[percentile]),
                            xref=f"x{col_idx}",
                            yref=f"y{col_idx}",
                        )
                    )

    # Add initial annotations for first month
    fig.update_layout(annotations=annotations_by_month[year_months[0]])

    # Create buttons for month selection
    buttons = []

    for i, year_month in enumerate(year_months):
        # Create visibility list for traces
        visibility = []
        for j, ym in enumerate(year_months):
            for zone in zones:
                visibility.append(ym == year_month)
                # Add visibility for each percentile line (4 percentiles per zone)
                for _ in percentiles:
                    visibility.append(ym == year_month)

        # Convert to pretty format for display
        month_obj = datetime.strptime(year_month, "%Y-%m")
        pretty_label = month_obj.strftime("%B %Y")

        buttons.append(
            dict(
                label=pretty_label,
                method="update",
                args=[
                    {"visible": visibility},
                    {"annotations": annotations_by_month[year_month],
                     "title": f"Load Distribution - {pretty_label}"}
                ]
            )
        )

    # Update layout with dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ],
        title=f"Load Distribution - {datetime.strptime(year_months[0], '%Y-%m').strftime('%B %Y')}",
        height=500,
        template="plotly_white",
        showlegend=True
    )

    # Update x-axes labels
    fig.update_xaxes(title_text="Load (MW)")
    fig.update_yaxes(title_text="Density")

    # Save as HTML
    fig.write_html(output_file)

    return fig


def create_load_profile_figure_percent(df, year_months, zones=None, percentile_data=None,
                                       output_file="load_profile_percentiles.html"):
    """
    Create an interactive load profile figure with month selector showing all years for that month.
    Shows 4 separate subplots, one for each zone.
    X-axis shows day and hour of month.
    Each year appears in a different colour.
    """

    if zones is None:
        zones = ["Connecticut", "Western Mass", "New Hampshire", "Total (NH+CT+WMass)"]

    # Define percentile colors
    percentile_colors = {90: "green", 95: "blue", 97.5: "orange", 99: "red"}

    # Get the actual timestamp column name (first column)
    timestamp_col = df.columns[0]

    # Extract unique months and add day/hour columns
    # Convert to UTC first to get consistent hour values
    df_with_month = df.with_columns(
        pl.col(timestamp_col).dt.convert_time_zone("UTC").alias("tstamp_utc")
    ).with_columns(
        pl.col("year_month").str.split("-").list.get(0).alias("year"),
        pl.col("year_month").str.split("-").list.get(1).alias("month"),
        pl.col("tstamp_utc").dt.day().alias("day"),
        pl.col("tstamp_utc").dt.hour().alias("hour"),
        (pl.col("tstamp_utc").dt.day() + pl.col("tstamp_utc").dt.hour() / 24).alias("day_hour")
    )

    unique_months = sorted(df_with_month.select("month").unique().to_series().to_list())
    unique_years = sorted(df_with_month.select("year").unique().to_series().to_list())

    # Create a color palette for years
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"
    ]
    year_colors = {year: colors[i % len(colors)] for i, year in enumerate(unique_years)}

    # Create figure with subplots (2x2 grid)
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=zones,
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Store trace info for visibility management
    trace_info = []

    # Create traces for each month, year, and zone
    for month in unique_months:
        month_data = df_with_month.filter(pl.col("month") == month)

        for zone_idx, zone in enumerate(zones, 1):
            row = (zone_idx - 1) // 2 + 1
            col = (zone_idx - 1) % 2 + 1

            zone_data = month_data.filter(pl.col("zone") == zone)

            # Group by year and add separate trace for each year
            for year in unique_years:
                year_zone_data = (zone_data
                                  .filter(pl.col("year") == year)
                                  .filter(pl.col("value").is_not_null())
                                  .sort(timestamp_col)
                                  .unique(subset=[timestamp_col], keep="first")
                                  )

                if len(year_zone_data) > 0:
                    x_vals = year_zone_data.select("day_hour").to_series().to_list()
                    y_vals = year_zone_data.select("value").to_series().to_list()

                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode="lines",
                            name=f"{year}",
                            visible=(month == unique_months[0]),
                            line=dict(color=year_colors[year]),
                            hovertemplate=f"<b>{zone} ({year})</b><br>Day: %{{customdata[0]}}, Hour: %{{customdata[1]}}<br>Load: %{{y:.2f}} MW<extra></extra>",
                            customdata=year_zone_data.select(["day", "hour"]).to_numpy(),
                            legendgroup=f"{year}",
                            showlegend=(zone_idx == 1),
                        ),
                        row=row, col=col
                    )
                    trace_info.append((month, zone, year, False, None))

    # Get x-axis range for each month
    x_ranges = {}
    for month in unique_months:
        month_data = df_with_month.filter(pl.col("month") == month)
        day_hour_values = month_data.select("day_hour").to_series()
        if len(day_hour_values) > 0:
            x_ranges[month] = (day_hour_values.min(), day_hour_values.max())

    # Add percentile lines as traces
    if percentile_data is not None and isinstance(percentile_data, pl.DataFrame):
        for month in unique_months:
            month_percentiles = percentile_data.filter(pl.col("month") == month)
            x_min, x_max = x_ranges.get(month, (1, 31))

            for row_data in month_percentiles.iter_rows(named=True):
                zone = row_data['zone']
                zone_idx = zones.index(zone) + 1
                row = (zone_idx - 1) // 2 + 1
                col = (zone_idx - 1) % 2 + 1

                for percentile, col_name in [(90, 'p90'), (95, 'p95'), (97.5, 'p97_5'), (99, 'p99')]:
                    value = row_data[col_name]

                    if value is not None and not (isinstance(value, float) and (value != value)):
                        fig.add_trace(
                            go.Scatter(
                                x=[x_min, x_max],
                                y=[value, value],
                                mode="lines",
                                name=f"{percentile}th",
                                visible=(month == unique_months[0]),
                                line=dict(
                                    color=percentile_colors.get(percentile, "gray"),
                                    width=2,
                                    dash="dash"
                                ),
                                hovertemplate=f"{percentile}th: {value:.0f}<extra></extra>",
                                showlegend=False,
                            ),
                            row=row, col=col
                        )
                        trace_info.append((month, zone, None, True, percentile))

    # Create buttons for month selection
    buttons = []
    month_names = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]

    for selected_month in unique_months:
        visibility = [
            (info[0] == selected_month)
            for info in trace_info
        ]

        month_name = month_names[int(selected_month) - 1]

        buttons.append(
            dict(
                label=month_name,
                method="update",
                args=[
                    {
                        "visible": visibility,
                    },
                    {
                        "title": f"Load Profile - {month_name} (All Years)"
                    }
                ]
            )
        )

    # Update layout with dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ],
        title=f"Load Profile - {month_names[int(unique_months[0]) - 1]} (All Years)",
        height=1000,
        template="plotly_white",
        showlegend=True,
        hovermode="closest",
        legend=dict(
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top"
        )
    )

    # Update all x and y axes
    fig.update_xaxes(title_text="Day of Month (with Hour)")
    fig.update_yaxes(title_text="Load (MW)")

    # Save as HTML
    fig.write_html(output_file)

    return fig


def create_load_percentile_figure(df, year_months, zones=None, percentile_data=None,
                                       output_file="load_percentiles.html",
                                       percentile_type="average"):
    """
    Create an interactive load profile figure with month selector showing all years for that month.
    Shows 4 separate subplots, one for each zone.
    X-axis shows day and hour of month.
    Each year appears in a different colour.
    Percentile lines show average or median values across years.

    Parameters:
    -----------
    df : pl.DataFrame
        Load data with timestamp, value, zone, year_month columns
    year_months : list
        List of year-month strings
    zones : list, optional
        List of zone names
    percentile_data : pl.DataFrame, optional
        DataFrame with columns: month, zone, percentile, average, median
    output_file : str
        Output HTML file path
    percentile_type : str
        Either "average" or "median" - which value to display for percentile lines
    """

    if zones is None:
        zones = ["Connecticut", "Western Mass", "New Hampshire", "Total (NH+CT+WMass)"]

    if percentile_type not in ["average", "median"]:
        raise ValueError("percentile_type must be 'average' or 'median'")

    # Define percentile colors
    percentile_colors = {90: "green", 95: "blue", 97.5: "orange", 99: "red"}

    # Get the actual timestamp column name (first column)
    timestamp_col = df.columns[0]

    # Extract unique months and add day/hour columns
    # Convert to UTC first to get consistent hour values
    df_with_month = df.with_columns(
        pl.col(timestamp_col).dt.convert_time_zone("UTC").alias("tstamp_utc")
    ).with_columns(
        pl.col("year_month").str.split("-").list.get(0).alias("year"),
        pl.col("year_month").str.split("-").list.get(1).alias("month"),
        pl.col("tstamp_utc").dt.day().alias("day"),
        pl.col("tstamp_utc").dt.hour().alias("hour"),
        (pl.col("tstamp_utc").dt.day() + pl.col("tstamp_utc").dt.hour() / 24).alias("day_hour")
    )

    unique_months = sorted(df_with_month.select("month").unique().to_series().to_list())
    unique_years = sorted(df_with_month.select("year").unique().to_series().to_list())

    # Create a color palette for years
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"
    ]
    year_colors = {year: colors[i % len(colors)] for i, year in enumerate(unique_years)}

    # Create figure with subplots (2x2 grid)
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=zones,
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Store trace info for visibility management
    trace_info = []

    # Create traces for each month, year, and zone
    for month in unique_months:
        month_data = df_with_month.filter(pl.col("month") == month)

        for zone_idx, zone in enumerate(zones, 1):
            row = (zone_idx - 1) // 2 + 1
            col = (zone_idx - 1) % 2 + 1

            zone_data = month_data.filter(pl.col("zone") == zone)

            # Group by year and add separate trace for each year
            for year in unique_years:
                year_zone_data = (zone_data
                                  .filter(pl.col("year") == year)
                                  .filter(pl.col("value").is_not_null())
                                  .sort(timestamp_col)
                                  .unique(subset=[timestamp_col], keep="first")
                                  )

                if len(year_zone_data) > 0:
                    x_vals = year_zone_data.select("day_hour").to_series().to_list()
                    y_vals = year_zone_data.select("value").to_series().to_list()

                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode="lines",
                            name=f"{year}",
                            visible=(month == unique_months[0]),
                            line=dict(color=year_colors[year]),
                            hovertemplate=f"<b>{zone} ({year})</b><br>Day: %{{customdata[0]}}, Hour: %{{customdata[1]}}<br>Load: %{{y:.2f}} MW<extra></extra>",
                            customdata=year_zone_data.select(["day", "hour"]).to_numpy(),
                            legendgroup=f"{year}",
                            showlegend=(zone_idx == 1),
                        ),
                        row=row, col=col
                    )
                    trace_info.append((month, zone, year, False, None))

    # Get x-axis range for each month
    x_ranges = {}
    for month in unique_months:
        month_data = df_with_month.filter(pl.col("month") == month)
        day_hour_values = month_data.select("day_hour").to_series()
        if len(day_hour_values) > 0:
            x_ranges[month] = (day_hour_values.min(), day_hour_values.max())

    # Add percentile lines as traces
    if percentile_data is not None and isinstance(percentile_data, pl.DataFrame):
        for month in unique_months:
            month_percentiles = percentile_data.filter(pl.col("month") == month)
            x_min, x_max = x_ranges.get(month, (1, 31))

            for row_data in month_percentiles.iter_rows(named=True):
                zone = row_data['zone']
                percentile_str = row_data['percentile']

                # Convert percentile string to float
                percentile = float(percentile_str)

                zone_idx = zones.index(zone) + 1
                row = (zone_idx - 1) // 2 + 1
                col = (zone_idx - 1) % 2 + 1

                # Get the value based on percentile_type
                value = row_data[percentile_type]

                if value is not None and not (isinstance(value, float) and (value != value)):
                    fig.add_trace(
                        go.Scatter(
                            x=[x_min, x_max],
                            y=[value, value],
                            mode="lines",
                            name=f"{percentile_str}th",
                            visible=(month == unique_months[0]),
                            line=dict(
                                color=percentile_colors.get(percentile, "gray"),
                                width=2,
                                dash="dash"
                            ),
                            hovertemplate=f"{percentile_str}th ({percentile_type}): {value:.0f}<extra></extra>",
                            showlegend=False,
                        ),
                        row=row, col=col
                    )
                    trace_info.append((month, zone, None, True, percentile))

    # Create buttons for month selection
    buttons = []
    month_names = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]

    for selected_month in unique_months:
        visibility = [
            (info[0] == selected_month)
            for info in trace_info
        ]

        month_name = month_names[int(selected_month) - 1]

        buttons.append(
            dict(
                label=month_name,
                method="update",
                args=[
                    {
                        "visible": visibility,
                    },
                    {
                        "title": f"Load Profile - {month_name} (All Years)"
                    }
                ]
            )
        )

    # Update layout with dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ],
        title=f"Load Profile - {month_names[int(unique_months[0]) - 1]} (All Years) - {percentile_type.capitalize()} Percentiles",
        height=1000,
        template="plotly_white",
        showlegend=True,
        hovermode="closest",
        legend=dict(
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top"
        )
    )

    # Update all x and y axes
    fig.update_xaxes(title_text="Day of Month (with Hour)")
    fig.update_yaxes(title_text="Load (MW)")

    # Save as HTML
    fig.write_html(output_file)

    return fig


def create_peak_calendar_from_percentiles(df, percentiles_df, zones=None,
                                          percentile_threshold=95,
                                          output_file="peak_calendar_percentiles.html"):
    """
    Create a calendar heatmap showing which days are most likely to have peaks
    based on percentile thresholds from calculate_percentiles_years.
    With month selector dropdown.

    Parameters:
    -----------
    df : pl.DataFrame
        Load data with timestamp, value, zone, year_month columns
    percentiles_df : pl.DataFrame
        Output from calculate_percentiles_years with columns:
        year, month, zone, p90, p95, p97_5, p99
    zones : list, optional
        List of zone names
    percentile_threshold : int or float
        Which percentile to use as threshold (90, 95, 97.5, 99)
    output_file : str
        Output HTML file path

    Returns:
    --------
    fig : plotly figure
        Calendar heatmap figure
    """

    if zones is None:
        zones = ["Connecticut", "Western Mass", "New Hampshire", "Total (NH+CT+WMass)"]

    # Get timestamp column name
    timestamp_col = df.columns[0]

    # Extract date information - convert to UTC to ensure consistent weekday
    df_with_date = df.with_columns(
        pl.col(timestamp_col).dt.convert_time_zone("UTC").alias("tstamp_utc")
    ).with_columns(
        pl.col("tstamp_utc").dt.date().alias("date"),
        pl.col("tstamp_utc").dt.year().cast(pl.Utf8).alias("year"),
        pl.col("tstamp_utc").dt.month().cast(pl.Utf8).str.zfill(2).alias("month"),
        pl.col("tstamp_utc").dt.day().alias("day"),
        (pl.col("tstamp_utc").dt.weekday() - 1).alias("weekday")  # Convert 1-7 to 0-6
    )

    # Get daily max for each day and zone
    daily_max = df_with_date.group_by(["date", "zone", "year", "month", "day", "weekday"]).agg(
        pl.col("value").max().alias("daily_max")
    )

    # Convert percentile threshold to column name
    percentile_col = f"p{percentile_threshold}".replace(".", "_")

    # Get the threshold for each year-month-zone combination
    threshold_data = percentiles_df.select(["year", "month", "zone", percentile_col]).rename(
        {percentile_col: "threshold"}
    )

    # Join daily max with thresholds
    daily_max = daily_max.join(
        threshold_data,
        on=["year", "month", "zone"]
    )

    # Determine if day is a peak day
    daily_max = daily_max.with_columns(
        (pl.col("daily_max") >= pl.col("threshold")).alias("is_peak")
    )

    # Get unique months
    unique_months = sorted(daily_max.select("month").unique().to_series().to_list())

    # Create subplots for each zone
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=zones,
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )

    month_names = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]

    # Store trace info for visibility management
    trace_info = []

    for zone_idx, zone in enumerate(zones, 1):
        row = (zone_idx - 1) // 2 + 1
        col = (zone_idx - 1) % 2 + 1

        # Filter for this zone
        zone_data = daily_max.filter(pl.col("zone") == zone)

        # Get unique year-months
        unique_year_months = zone_data.select(
            (pl.col("year") + "-" + pl.col("month")).alias("year_month")
        ).unique().sort("year_month").to_series().to_list()

        # Plot each year-month
        for year_month in unique_year_months:
            year, month = year_month.split("-")

            # Filter for this year-month
            ym_data = zone_data.filter(
                (pl.col("year") == year) & (pl.col("month") == month)
            )

            # Calculate week number using day of month
            ym_data = ym_data.with_columns(
                ((pl.col("day") - 1) // 7).alias("week")
            )

            # Separate peak and normal days
            peak_data = ym_data.filter(pl.col("is_peak") == True)
            normal_data = ym_data.filter(pl.col("is_peak") == False)

            # Plot normal days (blue)
            if len(normal_data) > 0:
                x_vals = normal_data.select("weekday").to_numpy().flatten().tolist()
                y_vals = normal_data.select("week").to_numpy().flatten().tolist()
                day_vals = normal_data.select("day").to_numpy().flatten().tolist()
                date_vals = normal_data.select("date").to_numpy().flatten().tolist()
                max_vals = normal_data.select("daily_max").to_numpy().flatten().tolist()
                threshold_vals = normal_data.select("threshold").to_numpy().flatten().tolist()

                # Create customdata for hover
                customdata = np.column_stack((
                    [str(d) for d in date_vals],
                    max_vals,
                    threshold_vals
                ))

                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode='markers+text',
                        marker=dict(size=20, color='#1f77b4'),
                        text=day_vals,
                        textposition='middle center',
                        textfont=dict(color='white', size=10),
                        hovertemplate='<b>%{customdata[0]}</b><br>Max: %{customdata[1]:.0f} MW<br>Threshold: %{customdata[2]:.0f} MW<extra></extra>',
                        customdata=customdata,
                        showlegend=False,
                        name=f"{zone} - {year_month}",
                        visible=(month == unique_months[0])
                    ),
                    row=row, col=col
                )
                trace_info.append((month, zone, "normal"))

            # Plot peak days (red)
            if len(peak_data) > 0:
                x_vals = peak_data.select("weekday").to_numpy().flatten().tolist()
                y_vals = peak_data.select("week").to_numpy().flatten().tolist()
                day_vals = peak_data.select("day").to_numpy().flatten().tolist()
                date_vals = peak_data.select("date").to_numpy().flatten().tolist()
                max_vals = peak_data.select("daily_max").to_numpy().flatten().tolist()
                threshold_vals = peak_data.select("threshold").to_numpy().flatten().tolist()

                # Create customdata for hover
                customdata = np.column_stack((
                    [str(d) for d in date_vals],
                    max_vals,
                    threshold_vals
                ))

                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode='markers+text',
                        marker=dict(size=20, color='#d62728'),
                        text=day_vals,
                        textposition='middle center',
                        textfont=dict(color='white', size=10),
                        hovertemplate='<b>%{customdata[0]}</b><br>Max: %{customdata[1]:.0f} MW<br>Threshold: %{customdata[2]:.0f} MW<br>PEAK DAY<extra></extra>',
                        customdata=customdata,
                        showlegend=False,
                        name=f"{zone} - {year_month} (Peak)",
                        visible=(month == unique_months[0])
                    ),
                    row=row, col=col
                )
                trace_info.append((month, zone, "peak"))

    # Create buttons for month selection
    buttons = []

    for selected_month in unique_months:
        visibility = [
            (info[0] == selected_month)
            for info in trace_info
        ]

        month_name = month_names[int(selected_month) - 1]

        buttons.append(
            dict(
                label=month_name,
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"Peak Days Calendar - {month_name} (>{percentile_threshold}th Percentile)"}
                ]
            )
        )

    # Update layout with dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ],
        title=f"Peak Days Calendar - {month_names[int(unique_months[0]) - 1]} (>{percentile_threshold}th Percentile)",
        height=1000,
        template="plotly_white",
        showlegend=False
    )

    # Update axes
    fig.update_xaxes(title_text="Day of Week",
                     tickvals=[1, 2, 3, 4, 5, 6],
                     ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                     range=[-0.5, 6.5])
    fig.update_yaxes(title_text="Week Number", autorange="reversed")

    fig.write_html(output_file)

    return fig


def create_prediction_figure(predictions_df, month, year, zone):
    """
    Create interactive Plotly figure for predictions
    Works directly with Polars without PyArrow
    Includes annotation for peak demand
    """
    if predictions_df is None or predictions_df.shape[0] == 0:
        print(f"Error: No predictions dataframe provided")
        return None

    # Filter predictions
    pred_month = predictions_df.filter(
        (pl.col("zone") == zone) &
        (pl.col("timestamp").dt.month() == month) &
        (pl.col("timestamp").dt.year() == year)
    )

    if pred_month.shape[0] == 0:
        print(f"Warning: No predictions found for {zone} in {month}/{year}")
        return None

    # Convert to lists instead of pandas
    timestamps = pred_month.select("timestamp").to_series().to_list()
    predicted_mw = pred_month.select("predicted_mw").to_series().to_list()
    p10 = pred_month.select("p10").to_series().to_list()
    p25 = pred_month.select("p25").to_series().to_list()
    p75 = pred_month.select("p75").to_series().to_list()
    p90 = pred_month.select("p90").to_series().to_list()
    historical_avg = pred_month.select("historical_avg").to_series().to_list()
    historical_min = pred_month.select("historical_min").to_series().to_list()
    historical_max = pred_month.select("historical_max").to_series().to_list()

    # Find peak prediction
    peak_idx = predicted_mw.index(max(predicted_mw))
    peak_value = predicted_mw[peak_idx]
    peak_timestamp = timestamps[peak_idx]

    fig = go.Figure()

    # Add percentile bands (10th-90th)
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=p90,
        fill=None,
        mode='lines',
        line_color='rgba(0,0,255,0)',
        showlegend=False,
        name='p90'
    ))

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=p10,
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,255,0)',
        name='10th-90th Percentile',
        fillcolor='rgba(0,100,200,0.2)'
    ))

    # Add percentile bands (25th-75th)
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=p75,
        fill=None,
        mode='lines',
        line_color='rgba(0,0,255,0)',
        showlegend=False,
        name='p75'
    ))

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=p25,
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,255,0)',
        name='25th-75th Percentile',
        fillcolor='rgba(0,100,200,0.3)'
    ))

    # Add historical min/max as a shaded region
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=historical_max,
        fill=None,
        mode='lines',
        line_color='rgba(200,200,200,0)',
        showlegend=False,
        name='Historical Max'
    ))

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=historical_min,
        fill='tonexty',
        mode='lines',
        line_color='rgba(200,200,200,0)',
        name='Historical Min/Max Range',
        fillcolor='rgba(200,200,200,0.1)'
    ))

    # Add historical average
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=historical_avg,
        mode='lines',
        name='Historical Average',
        line=dict(color='green', width=1, dash='dash')
    ))

    # Add prediction line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=predicted_mw,
        mode='lines',
        name='Predicted',
        line=dict(color='red', width=2)
    ))

    # Add peak point marker
    fig.add_trace(go.Scatter(
        x=[peak_timestamp],
        y=[peak_value],
        mode='markers',
        name='Peak Demand',
        marker=dict(
            size=12,
            color='darkred',
            symbol='star',
            line=dict(color='yellow', width=2)
        ),
        showlegend=True
    ))

    # Add annotation for peak
    fig.add_annotation(
        x=peak_timestamp,
        y=peak_value,
        text=f"<b>Peak: {peak_value:.0f} MW</b><br>{peak_timestamp.strftime('%b %d, %H:%M')}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='darkred',
        ax=-50,
        ay=-50,
        bgcolor='rgba(255, 255, 200, 0.8)',
        bordercolor='darkred',
        borderwidth=2,
        font=dict(size=12, color='darkred'),
        align='center'
    )

    fig.update_layout(
        title=f"Predicted MW Usage - {zone} - {datetime(year, month, 1).strftime('%B %Y')}",
        xaxis_title="Date",
        yaxis_title="Megawatts",
        hovermode='x unified',
        template='plotly_white',
        height=1000,
        width=2400
    )

    return fig


def create_all_visualizations(predictions_df, month, year, zones, output_dir="predict/figs/"):
    """
    Create individual zone figures and summary figure for a specific month

    Parameters:
    - predictions_df: Polars dataframe with predictions
    - month: Month number (1-12)
    - year: Year to visualize
    - zones: List of zone names
    - output_dir: Directory to save HTML files
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    month_name = datetime(year, month, 1).strftime('%B %Y')

    print("\n" + "=" * 70)
    print(f"Creating Visualizations for {month_name}")
    print("=" * 70)

    # Create individual zone figures
    print(f"\nCreating individual zone figures...")
    for zone in zones:
        try:
            fig = create_prediction_figure(predictions_df, month=month, year=year, zone=zone)
            if fig is not None:
                filename = f"prediction_{year}_{month:02d}_{zone.replace(' ', '_').replace('(', '').replace(')', '').lower()}.html"
                filepath = os.path.join(output_dir, filename)
                fig.write_html(filepath)
                print(f"  Saved: {filename}")
            else:
                print(f"  No data for {zone}")
        except Exception as e:
            print(f"  Error creating figure for {zone}: {str(e)}")

    # Create summary figure for all zones
    print(f"\nCreating summary figure for all zones...")
    try:
        fig_summary = create_summary_figure(predictions_df, month, year, zones)

        if fig_summary is not None:
            summary_filename = f"prediction_{year}_{month:02d}_all_zones_summary.html"
            summary_filepath = os.path.join(output_dir, summary_filename)
            fig_summary.write_html(summary_filepath)
            print(f"  Saved: {summary_filename}")
    except Exception as e:
        print(f"  Error creating summary figure: {str(e)}")

    print("\n" + "=" * 70)
    print(f" Visualizations for {month_name} saved to {output_dir}")
    print("=" * 70)


def create_summary_figure(predictions_df, month, year, zones):
    """
    Create a summary figure showing all zones with peak annotations

    Parameters:
    - predictions_df: Polars dataframe with predictions
    - month: Month number (1-12)
    - year: Year to visualize
    - zones: List of zone names

    Returns:
    - Plotly figure object
    """
    fig_summary = go.Figure()
    colors = ['red', 'blue', 'green', 'purple']

    month_name = datetime(year, month, 1).strftime('%B %Y')

    for idx, zone in enumerate(zones):
        pred_zone = predictions_df.filter(
            (pl.col("zone") == zone) &
            (pl.col("timestamp").dt.month() == month) &
            (pl.col("timestamp").dt.year() == year)
        )

        if pred_zone.shape[0] > 0:
            timestamps = pred_zone.select("timestamp").to_series().to_list()
            predicted_mw = pred_zone.select("predicted_mw").to_series().to_list()

            # Find peak for this zone
            peak_idx = predicted_mw.index(max(predicted_mw))
            peak_value = predicted_mw[peak_idx]
            peak_timestamp = timestamps[peak_idx]

            # Add prediction line
            fig_summary.add_trace(go.Scatter(
                x=timestamps,
                y=predicted_mw,
                mode='lines',
                name=zone,
                line=dict(width=2, color=colors[idx])
            ))

            # Add peak marker
            fig_summary.add_trace(go.Scatter(
                x=[peak_timestamp],
                y=[peak_value],
                mode='markers+text',
                name=f'{zone} Peak',
                marker=dict(
                    size=10,
                    color=colors[idx],
                    symbol='star',
                    line=dict(color='yellow', width=1)
                ),
                text=[f'{peak_value:.0f} MW'],
                textposition='top center',
                textfont=dict(size=10, color=colors[idx]),
                showlegend=True
            ))

    fig_summary.update_layout(
        title=f"Predicted MW Usage - All Zones - {month_name}",
        xaxis_title="Date",
        yaxis_title="Megawatts",
        hovermode='x unified',
        template='plotly_white',
        height=700,
        width=1400,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig_summary
