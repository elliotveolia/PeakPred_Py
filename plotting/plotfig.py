import polars as pl
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
    df_with_month = df.with_columns(
        pl.col("year_month").str.split("-").list.get(0).alias("year"),  # Extract year
        pl.col("year_month").str.split("-").list.get(1).alias("month"),
        pl.col(timestamp_col).dt.day().alias("day"),
        pl.col(timestamp_col).dt.hour().alias("hour"),
        (pl.col(timestamp_col).dt.day() + pl.col(timestamp_col).dt.hour() / 24).alias("day_hour")
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
    trace_info = []  # List of (month, zone, year, is_percentile, percentile_value)

    # Create traces for each month, year, and zone
    for month in unique_months:
        month_data = df_with_month.filter(pl.col("month") == month)

        for zone_idx, zone in enumerate(zones, 1):
            row = (zone_idx - 1) // 2 + 1
            col = (zone_idx - 1) % 2 + 1

            zone_data = month_data.filter(pl.col("zone") == zone).sort(timestamp_col)

            # Group by year and add separate trace for each year
            for year in unique_years:
                year_zone_data = zone_data.filter(pl.col("year") == year)

                if len(year_zone_data) > 0:  # Only add if data exists
                    fig.add_trace(
                        go.Scatter(
                            x=year_zone_data.select("day_hour").to_series(),
                            y=year_zone_data.select("value").to_series(),
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

    # Add percentile lines as traces (not hlines) for ALL months
    if percentile_data is not None and isinstance(percentile_data, pl.DataFrame):
        for month in unique_months:
            month_percentiles = percentile_data.filter(pl.col("month") == month)
            x_min, x_max = x_ranges.get(month, (1, 31))

            for row_data in month_percentiles.iter_rows(named=True):
                zone = row_data['zone']
                zone_idx = zones.index(zone) + 1
                row = (zone_idx - 1) // 2 + 1
                col = (zone_idx - 1) % 2 + 1

                # Add traces for each percentile
                for percentile, col_name in [(90, 'p90'), (95, 'p95'), (97.5, 'p97_5'), (99, 'p99')]:
                    value = row_data[col_name]

                    if value is not None:
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
        # Create visibility list based on trace_info
        visibility = [
            (info[0] == selected_month)  # Show if trace's month matches selected month
            for info in trace_info
        ]

        # Convert month number to name
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
