#######################################################################################################################
# Imports #
#######################################################################################################################
import os
import polars as pl
import calendar
import plotting.plotfig as plot
import functions as funct

from ice_data_py import cds, hist3
from datetime import datetime, date
from datetime import datetime, timedelta

#######################################################################################################################
# Data Import #0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#######################################################################################################################

# Define start and end time
t1 = datetime(2021, 6, 1, 0)
t2 = datetime(2025, 12, 31, 23)

print("=" * 70)
print("Loading hourly generation data from CDS")
print("Time Period:", t1, " to ", t2)
print("=" * 70)

## Load in hourly zones with zone-specific temperatures
print("\nLoading demand data...")
ct_hourly = cds.fetch_all([cds.QuerySpec(ms="NEISO", mn="load_zone._z_connecticut", mp="realtime_hourly_demand")], t1, t2)
wcmass_hourly = cds.fetch_all([cds.QuerySpec(ms="NEISO", mn="load_zone._z_wcmass", mp="realtime_hourly_demand")], t1, t2)
nh_hourly = cds.fetch_all([cds.QuerySpec(ms="NEISO", mn="load_zone._z_newhampshire", mp="realtime_hourly_demand")], t1, t2)

print("Loading zone-specific temperature data...")
ct_temp_hourly = cds.fetch_all([cds.QuerySpec(ms="NOAA-Forecast", mn="CT-Groton", mp="temperature[degF]")], t1, t2)
wcmass_temp_hourly = cds.fetch_all([cds.QuerySpec(ms="NOAA-Forecast", mn="MA-Worcester", mp="temperature[degF]")], t1, t2)
nh_temp_hourly = cds.fetch_all([cds.QuerySpec(ms="dev4-TWC-Forecasts", mn="VT.Georgia.coordinates", mp="temperature_F")], t1, t2)

print("Loading zone-specific dew point data...")
ct_dewpoint_hourly = cds.fetch_all([cds.QuerySpec(ms="dev4-TWC-Forecasts", mn="MA.Northampton.coordinates", mp="dewPoint_F")], t1, t2)
wcmass_dewpoint_hourly = cds.fetch_all([cds.QuerySpec(ms="dev4-TWC-Forecasts", mn="MA.Worcester.coordinates", mp="dewPoint_F")], t1, t2)
nh_dewpoint_hourly = cds.fetch_all([cds.QuerySpec(ms="dev4-TWC-Forecasts", mn="VT.Georgia.coordinates", mp="dewPoint_F")], t1, t2)

# Convert to Polars DataFrames
ct_df = pl.DataFrame(ct_hourly)
wcmass_df = pl.DataFrame(wcmass_hourly)
nh_df = pl.DataFrame(nh_hourly)

ct_temp_df = pl.DataFrame(ct_temp_hourly)
wcmass_temp_df = pl.DataFrame(wcmass_temp_hourly)
nh_temp_df = pl.DataFrame(nh_temp_hourly)

ct_dewpoint_df = pl.DataFrame(ct_dewpoint_hourly)
wcmass_dewpoint_df = pl.DataFrame(wcmass_dewpoint_hourly)
nh_dewpoint_df = pl.DataFrame(nh_dewpoint_hourly)

print(f"CT: {len(ct_hourly)} demand records, {len(ct_temp_hourly)} temp records, {len(ct_dewpoint_hourly)} dewpoint records")
print(f"WMass: {len(wcmass_hourly)} demand records, {len(wcmass_temp_hourly)} temp records, {len(wcmass_dewpoint_hourly)} dewpoint records")
print(f"NH: {len(nh_hourly)} demand records, {len(nh_temp_hourly)} temp records, {len(nh_dewpoint_hourly)} dewpoint records")


#######################################################################################################################
# Process Zone Data #
#######################################################################################################################
print("\nProcessing zone data...")
ct_processed = funct.process_zone_data_with_dewpoint(ct_df, ct_temp_df, ct_dewpoint_df, "Connecticut")
wcmass_processed = funct.process_zone_data_with_dewpoint(wcmass_df, wcmass_temp_df, wcmass_dewpoint_df, "Western Mass")
nh_processed = funct.process_zone_data_with_dewpoint(nh_df, nh_temp_df, nh_dewpoint_df, "New Hampshire")

print(f"Connecticut: {ct_processed.shape[0]} records")
print(f"Western Mass: {wcmass_processed.shape[0]} records")
print(f"New Hampshire: {nh_processed.shape[0]} records")

# Combine all zones
df = pl.concat([ct_processed, wcmass_processed, nh_processed])

# Extract temporal features
df = df.with_columns(
    pl.col("timestamp").dt.strftime("%Y-%m").alias("year_month"),
    pl.col("timestamp").dt.hour().alias("hour"),
    pl.col("timestamp").dt.year().alias("year"),
    pl.col("timestamp").dt.month().alias("month"),
    pl.col("timestamp").dt.day().alias("day"),
    pl.col("timestamp").dt.weekday().alias("dayofweek"),
)

# Create combined total zone
df_combined = df.group_by("timestamp", "year_month", "hour", "year", "month", "day", "dayofweek").agg(
    pl.col("value").sum().alias("value"),
    pl.col("temperature").mean().alias("temperature"),
    pl.col("dewpoint").mean().alias("dewpoint"),
    pl.col("feels_like").mean().alias("feels_like")
).with_columns(
    pl.lit("Total (NH+CT+WMass)").alias("zone")
)

# Reorder columns to match df
df_combined = df_combined.select(df.columns)

# Add combined zone
df = pl.concat([df, df_combined])

# Get unique year-months sorted
year_months = sorted(df.select("year_month").unique().to_series().to_list())

print(f"\nFound {len(year_months)} months of data")
print("Months:")
for ym in year_months:
    month_obj = datetime.strptime(ym, "%Y-%m")
    pretty_month = month_obj.strftime("%B %Y")
    print(f"  - {pretty_month}")

print("\n" + "=" * 70)

#######################################################################################################################
# Create Figure #
#######################################################################################################################

# Usage - with combined zone
zones = ["Connecticut", "Western Mass", "New Hampshire", "Total (NH+CT+WMass)"]

percentile_months = funct.calculate_percentiles_months(df, year_months)

percentile_years = funct.calculate_percentiles_years(df, year_months)

aggregate_percentiles = funct.aggregate_percentiles_by_month(percentile_years)

monthly_stats = funct.get_monthly_avg_of_daily_extremes(df, year_months)

load_file = False
target_year = 2026

if load_file:
    #######################################################################################################################
    # Load Existing Predictions #
    #######################################################################################################################

    print("\n" + "=" * 70)
    print("Loading Predictions from File")
    print("=" * 70)

    predictions_file = f"predict/data/predictions_{target_year}.csv"

    try:
        predictions_combined = pl.read_csv(predictions_file)

        # Convert timestamp column to datetime
        predictions_combined = predictions_combined.with_columns(
            pl.col("timestamp").str.to_datetime().alias("timestamp")
        )

        print(f" Loaded predictions from: {predictions_file}")
        print(f"  Shape: {predictions_combined.shape}")
        print(f"  Columns: {predictions_combined.columns}")
        print(f"  Data types: {predictions_combined.dtypes}")
    except FileNotFoundError:
        print(f" Error: File not found - {predictions_file}")
        predictions_combined = None
    except Exception as e:
        print(f" Error loading file: {str(e)}")
        predictions_combined = None
else:
    #######################################################################################################################
    # Create Model #
    #######################################################################################################################

    print("\n" + "=" * 70)
    print("Engineering Features for ML Model")
    print("=" * 70)

    df_features = funct.engineer_features_with_weather_and_percentiles(
        df,
        percentile_months=percentile_months,
        monthly_stats=monthly_stats
    )

    print(f"Before drop_nulls: {df_features.shape}")
    print(f"Zones before: {df_features.select('zone').unique().to_series().to_list()}")

    df_features = df_features.drop_nulls(subset=["value", "temperature", "hour_avg"])

    print(f"After drop_nulls: {df_features.shape}")
    print(f"Zones after: {df_features.select('zone').unique().to_series().to_list()}")

    # Check each zone
    zones = ["Connecticut", "Western Mass", "New Hampshire"]
    for zone in zones:
        zone_count = df_features.filter(pl.col("zone") == zone).shape[0]
        print(f"  {zone}: {zone_count} records")

    print(f"\nColumns: {df_features.columns}")

    #######################################################################################################################
    # Train Models - Individual Zones Only #
    #######################################################################################################################

    print("\n" + "=" * 70)
    print("Training Models for Individual Zones Only")
    print("=" * 70)

    trained_models = {}
    zones_to_train = ["Connecticut", "Western Mass", "New Hampshire"]

    for zone in zones_to_train:
        df_zone = df_features.filter(pl.col("zone") == zone)
        # Use quantile regression instead - predicts 90th percentile
        trained_models[zone] = funct.train_zone_model_quantile(
            df_zone,
            zone,
            quantile=0.90  # Predict 90th percentile (peaks)
        )

    #######################################################################################################################
    # Generate Predictions #
    #######################################################################################################################

    print("\n" + "=" * 70)
    print("Generating Predictions for Individual Zones")
    print("=" * 70)

    target_year = 2026
    predictions_all = []

    # Define zone-specific weather locations
    zone_weather_locations = {
        "Connecticut": "CT-Groton",
        "Western Mass": "MA-Worcester",
        "New Hampshire": "VT.Georgia.coordinates",
    }

    # Define weather data sources (some zones use different APIs)
    zone_weather_sources = {
        "Connecticut": ("NOAA-Forecast", "temperature[degF]", "dewpoint[degF]"),
        "Western Mass": ("NOAA-Forecast", "temperature[degF]", "dewpoint[degF]"),
        "New Hampshire": ("dev4-TWC-Forecasts", "temperature_F", "dewpoint_F"),
    }

    for month in range(1, 7):
        month_name = datetime(target_year, month, 1).strftime('%B %Y')
        print(f"\n{month_name}:")

        # Define date range for this month
        t1_month = datetime(target_year, month, 1, 0)
        t2_month = datetime(target_year, month, 1) + timedelta(
            days=calendar.monthrange(target_year, month)[1]) - timedelta(seconds=1)
        t2_month = t2_month.replace(hour=23, minute=59, second=59)

        # Fetch zone-specific weather data for this month
        zone_weather_data = {}

        for zone in zones_to_train:
            try:
                location = zone_weather_locations[zone]
                ms, temp_mp, dewpoint_mp = zone_weather_sources[zone]

                # Fetch temperature
                weather_raw = cds.fetch_all(
                    [cds.QuerySpec(ms=ms, mn=location, mp=temp_mp)],
                    t1_month,
                    t2_month
                )

                # Fetch dewpoint
                dewpoint_raw = cds.fetch_all(
                    [cds.QuerySpec(ms=ms, mn=location, mp=dewpoint_mp)],
                    t1_month,
                    t2_month
                )

                # Convert to Polars DataFrames
                weather_df = pl.DataFrame(weather_raw)
                dewpoint_df = pl.DataFrame(dewpoint_raw)

                # Standardize column names
                weather_df = weather_df.rename({weather_df.columns[1]: "temperature"})
                dewpoint_df = dewpoint_df.rename({dewpoint_df.columns[1]: "dewpoint"})

                # Standardize timestamp
                temp_col = weather_df.columns[0]
                weather_df = weather_df.with_columns(
                    pl.col(temp_col).cast(pl.Datetime).alias("timestamp")
                )
                if temp_col != "timestamp":
                    weather_df = weather_df.drop(temp_col)

                dewpoint_col = dewpoint_df.columns[0]
                dewpoint_df = dewpoint_df.with_columns(
                    pl.col(dewpoint_col).cast(pl.Datetime).alias("timestamp")
                )
                if dewpoint_col != "timestamp":
                    dewpoint_df = dewpoint_df.drop(dewpoint_col)

                # Join temperature and dewpoint
                weather_df = weather_df.join(
                    dewpoint_df.select(["timestamp", "dewpoint"]),
                    on="timestamp",
                    how="left"
                )

                # Calculate feels-like temperature
                weather_df = weather_df.with_columns(
                    pl.struct(["temperature", "dewpoint"]).map_elements(
                        lambda x: funct.calculate_feels_like(x["temperature"], x["dewpoint"]),
                        return_dtype=pl.Float64
                    ).alias("feels_like")
                )

                zone_weather_data[zone] = weather_df

                # Calculate feels-like temperature
                weather_df = weather_df.with_columns(
                    pl.struct(["temperature", "dewpoint"]).map_elements(
                        lambda x: funct.calculate_feels_like(x["temperature"], x["dewpoint"]),
                        return_dtype=pl.Float64
                    ).alias("feels_like")
                )

                zone_weather_data[zone] = weather_df

                if weather_df.shape[0] > 0:
                    avg_temp = weather_df.select("temperature").mean().item()
                    min_temp = weather_df.select("temperature").min().item()
                    max_temp = weather_df.select("temperature").max().item()
                    avg_fl = weather_df.select("feels_like").mean().item()  # CHANGED FROM heat_index
                    print(
                        f"  {zone}: {weather_df.shape[0]} records | Avg Temp: {avg_temp:.1f}°F | Avg FL: {avg_fl:.1f}°F | Min: {min_temp:.1f}°F | Max: {max_temp:.1f}°F")
                else:
                    print(f"  {zone}: No weather data available")
                    zone_weather_data[zone] = None

            except Exception as e:
                print(f"  {zone}: Error fetching weather - {str(e)}")
                zone_weather_data[zone] = None

        # Generate predictions for individual zones only
        for zone in zones_to_train:
            try:
                pred_df = funct.predict_month_with_percentiles_and_weather(
                    zone=zone,
                    month=month,
                    year_to_predict=target_year,
                    df_input=df_features,
                    model_dict=trained_models[zone],
                    current_weather_data=zone_weather_data[zone],
                    percentile_data=percentile_months,
                    monthly_stats=monthly_stats
                )

                if pred_df is not None and pred_df.shape[0] > 0:
                    predictions_all.append(pred_df)
                    print(f"   {zone}: {pred_df.shape[0]} predictions")
                else:
                    print(f"  {zone}: No predictions generated")
            except Exception as e:
                print(f"  {zone}: {str(e)}")
                continue

    #######################################################################################################################
    # Save Individual Zone Predictions #
    #######################################################################################################################

    predictions_combined = funct.save_predictions(
        predictions_list=predictions_all,
        target_year=target_year,
        output_dir="predict/data/"
    )

    #######################################################################################################################
    # Calculate Total Zone as Sum of Individual Zones #
    #######################################################################################################################

    if predictions_combined is not None and predictions_combined.shape[0] > 0:
        # Check if Total zone already exists
        existing_zones = predictions_combined.select('zone').unique().to_series().to_list()

        if 'Total (NH+CT+WMass)' not in existing_zones:
            print("\n" + "=" * 70)
            print("Calculating Total Zone Predictions")
            print("=" * 70)

            total_predictions = []

            # Get all unique timestamps
            timestamps = predictions_combined.select("timestamp").unique().sort("timestamp").to_series().to_list()

            for ts in timestamps:
                try:
                    # Get predictions for each individual zone at this timestamp
                    ct_pred = predictions_combined.filter(
                        (pl.col("zone") == "Connecticut") &
                        (pl.col("timestamp") == ts)
                    )

                    wm_pred = predictions_combined.filter(
                        (pl.col("zone") == "Western Mass") &
                        (pl.col("timestamp") == ts)
                    )

                    nh_pred = predictions_combined.filter(
                        (pl.col("zone") == "New Hampshire") &
                        (pl.col("timestamp") == ts)
                    )

                    # Sum if all three zones have predictions
                    if ct_pred.shape[0] > 0 and wm_pred.shape[0] > 0 and nh_pred.shape[0] > 0:
                        ct_mw = ct_pred.select("predicted_mw").item()
                        wm_mw = wm_pred.select("predicted_mw").item()
                        nh_mw = nh_pred.select("predicted_mw").item()
                        total_mw = ct_mw + wm_mw + nh_mw

                        # Sum percentiles and historical bounds
                        p10 = ct_pred.select("p10").item() + wm_pred.select("p10").item() + nh_pred.select("p10").item()
                        p25 = ct_pred.select("p25").item() + wm_pred.select("p25").item() + nh_pred.select("p25").item()
                        p75 = ct_pred.select("p75").item() + wm_pred.select("p75").item() + nh_pred.select("p75").item()
                        p90 = ct_pred.select("p90").item() + wm_pred.select("p90").item() + nh_pred.select("p90").item()
                        hist_avg = ct_pred.select("historical_avg").item() + wm_pred.select(
                            "historical_avg").item() + nh_pred.select("historical_avg").item()
                        hist_min = ct_pred.select("historical_min").item() + wm_pred.select(
                            "historical_min").item() + nh_pred.select("historical_min").item()
                        hist_max = ct_pred.select("historical_max").item() + wm_pred.select(
                            "historical_max").item() + nh_pred.select("historical_max").item()

                        # PRESERVE TEMPERATURE DATA - Average across zones
                        ct_temp = ct_pred.select("temperature_used").item()
                        wm_temp = wm_pred.select("temperature_used").item()
                        nh_temp = nh_pred.select("temperature_used").item()
                        avg_temp = (ct_temp + wm_temp + nh_temp) / 3

                        # Average feels like
                        ct_fl = ct_pred.select("feels_like_used").item()
                        wm_fl = wm_pred.select("feels_like_used").item()
                        nh_fl = nh_pred.select("feels_like_used").item()
                        avg_fl = (ct_fl + wm_fl + nh_fl) / 3

                        # Use the most common temp_source
                        temp_sources = [
                            ct_pred.select("temp_source").item(),
                            wm_pred.select("temp_source").item(),
                            nh_pred.select("temp_source").item()
                        ]
                        temp_source = max(set(temp_sources), key=temp_sources.count)

                        # Average adjustment factors
                        ct_adj = ct_pred.select("temp_adjustment_factor").item()
                        wm_adj = wm_pred.select("temp_adjustment_factor").item()
                        nh_adj = nh_pred.select("temp_adjustment_factor").item()
                        avg_adj = (ct_adj + wm_adj + nh_adj) / 3

                        total_predictions.append({
                            'timestamp': ts,
                            'zone': 'Total (NH+CT+WMass)',
                            'hour': ct_pred.select("hour").item(),
                            'day': ct_pred.select("day").item(),
                            'predicted_mw': float(total_mw),
                            'p10': float(p10),
                            'p25': float(p25),
                            'p75': float(p75),
                            'p90': float(p90),
                            'historical_min': float(hist_min),
                            'historical_max': float(hist_max),
                            'historical_avg': float(hist_avg),
                            'temperature_used': float(avg_temp),
                            'feels_like_used': float(avg_fl),
                            'temp_source': temp_source,
                            'temp_adjustment_factor': float(avg_adj),
                        })
                except Exception as e:
                    continue

            # Add Total zone predictions to combined dataframe
            if total_predictions:
                total_df = pl.DataFrame(total_predictions)
                predictions_combined = pl.concat([predictions_combined, total_df])
                print(f"Total zone calculated: {total_df.shape[0]} predictions")

                #######################################################################################################################
                # Save Combined Predictions with Total Zone #
                #######################################################################################################################

                filepath = os.path.join("predict/data/", f"predictions_{target_year}.csv")
                predictions_combined.write_csv(filepath)
                print(f"Updated predictions saved to: {filepath}")

            else:
                print("Failed to calculate Total zone")
        else:
            print("\nTotal zone already exists in predictions")


#######################################################################################################################
# Create Visualizations #
#######################################################################################################################

if predictions_combined is not None and predictions_combined.shape[0] > 0:
    print("\n" + "=" * 70)
    print("Creating Prediction Visualizations")
    print("=" * 70)

    zones_to_plot = ["Connecticut", "Western Mass", "New Hampshire", "Total (NH+CT+WMass)"]

    # Create figures for each month
    for month in range(1, 7):
        month_name = datetime(target_year, month, 1).strftime('%B %Y')
        print(f"\n{month_name}:")

        for zone in zones_to_plot:
            try:
                fig = plot.create_prediction_figure(
                    predictions_df=predictions_combined,
                    month=month,
                    year=target_year,
                    zone=zone
                )

                if fig is not None:
                    # Create output directory if it doesn't exist

                    os.makedirs("predict/figs/", exist_ok=True)

                    # Save the figure
                    filename = f"prediction_{target_year}_{month:02d}_{zone.replace(' ', '_').replace('(', '').replace(')', '').lower()}.html"
                    filepath = os.path.join("predict/figs/", filename)
                    fig.write_html(filepath)
                    print(f"  {zone}: {filename}")
                else:
                    print(f"  {zone}: No data")

            except Exception as e:
                print(f"  {zone}: {str(e)}")

    print("\n" + "=" * 70)
    print("Prediction visualizations completed!")
    print("=" * 70)
else:
    print("\nNo predictions available to visualize")
