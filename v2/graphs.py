# plot_simple.py
"""
Simple plot of peak predictions, init peak, and load over time with month selector
"""

import polars as pl
import plotly.graph_objects as go
import os
from config import OUTPUT_DIR, OUTPUT_CSV_PEAK_PREDICT, OUTPUT_CSV_INIT_PEAK

# Load the CSV files
init_peak_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_INIT_PEAK)
peak_predict_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_PEAK_PREDICT)
load_path = os.path.join(OUTPUT_DIR, "load_data.csv")

print("Loading data...")
init_peak_df = pl.read_csv(init_peak_path)
peak_predict_df = pl.read_csv(peak_predict_path)
load_df = pl.read_csv(load_path)

# Extract months from data
init_peak_df = init_peak_df.with_columns(
    pl.col('date').cast(pl.Utf8).str.slice(0, 7).alias('month')
)
peak_predict_df = peak_predict_df.with_columns(
    pl.col('datetime').cast(pl.Utf8).str.slice(0, 7).alias('month')
)
load_df = load_df.with_columns(
    pl.col('datetime').cast(pl.Utf8).str.slice(0, 7).alias('month')
)

# Get unique months
months = sorted(init_peak_df['month'].unique().to_list())
print(f"Available months: {months}")

# Create figure
fig = go.Figure()

# Add traces for each month
for month in months:
    init_peak_month = init_peak_df.filter(pl.col('month') == month)
    peak_predict_month = peak_predict_df.filter(pl.col('month') == month)
    load_month = load_df.filter(pl.col('month') == month)

    # Add load_mw trace (left y-axis)
    fig.add_trace(go.Scatter(
        x=load_month['datetime'],
        y=load_month['load_mw'],
        mode='lines',
        name=f'Load MW - {month}',
        line=dict(color='green', width=2),
        yaxis='y1',
        visible=(month == months[0]),
        legendgroup=month
    ))

    # Add init_peak trace (left y-axis)
    fig.add_trace(go.Scatter(
        x=init_peak_month['date'],
        y=init_peak_month['init_peak'],
        mode='lines+markers',
        name=f'Initial Peak - {month}',
        line=dict(color='red', width=3),
        marker=dict(size=8),
        yaxis='y1',
        visible=(month == months[0]),
        legendgroup=month
    ))

    # Add predicted_peak_strength_actionable trace (right y-axis)
    fig.add_trace(go.Scatter(
        x=peak_predict_month['datetime'],
        y=peak_predict_month['predicted_peak_strength_actionable'],
        mode='lines',
        name=f'Peak Strength - {month}',
        line=dict(color='blue', width=2),
        yaxis='y2',
        visible=(month == months[0]),
        legendgroup=month
    ))

# Create visibility list for dropdown
visibility_list = []
for month in months:
    visibility = [False] * len(fig.data)
    for i, trace in enumerate(fig.data):
        if month in trace.name:
            visibility[i] = True
    visibility_list.append(visibility)

# Create dropdown buttons
buttons = []
for i, month in enumerate(months):
    buttons.append(
        dict(
            label=month,
            method='update',
            args=[
                {'visible': visibility_list[i]},
                {'title': f'Peak Predictions vs Initial Peak vs Load - {month}'}
            ]
        )
    )

# Update layout with dropdown and secondary y-axis
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
    title=f'Peak Predictions vs Initial Peak vs Load - {months[0]}',
    xaxis=dict(title='Date/Time'),
    yaxis=dict(
        title='MW (Load & Initial Peak)',
        title_font=dict(color='darkgreen'),
        tickfont=dict(color='darkgreen')
    ),
    yaxis2=dict(
        title='Peak Strength Actionable',
        title_font=dict(color='blue'),
        tickfont=dict(color='blue'),
        overlaying='y',
        side='right'
    ),
    hovermode='x unified',
    template='plotly_white',
    height=700,
    width=1200,
    legend=dict(
        x=0.0,
        y=0.99,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='black',
        borderwidth=1
    )
)

output_path = os.path.join(OUTPUT_DIR, "peak_comparison.html")
fig.write_html(output_path)
print(f"✓ Plot saved to: {output_path}")
