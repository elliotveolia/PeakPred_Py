#fig = create_load_profile_figure(df, year_months, zones=zones)
#fig.show()

#fig = create_load_distribution_figure(df, year_months,)
#fig.show()

#fig = create_load_profile_figure_percent(df, year_months, percentile_data=percentile_months)
#fig.show()

#fig_avg = create_load_percentile_figure(
#    df,
#    year_months,
#    percentile_data=aggregate_percentiles,
#    output_file="load_profile_average.html",
#    percentile_type="average"
#)

#fig_avg = create_load_percentile_figure(
#    df,
#    year_months,
#    percentile_data=aggregate_percentiles,
#    output_file="load_profile_median.html",
#    percentile_type="median"
#)

#fig = plot.create_load_profile_figure_minmax(
#    df=df,
#    year_months=year_months,
#    minmax_data=monthly_stats,
#    output_file="load_profile_minmax.html"
#)


#fig = plot.create_peak_calendar_from_percentiles(
#    df=df,
#    percentiles_df=percentile_years,
#    percentile_threshold=99,
#    output_file="peak_calendar_percentiles.html"
#)


def engineer_features(df_input):
    """
    Engineer features for the predictive model
    Works with existing df structure from CDS
    """
    df_feat = df_input.clone()

    # Identify timestamp column (first column from CDS)
    timestamp_col = df_feat.columns[0]

    # Ensure timestamp is datetime
    if df_feat[timestamp_col].dtype != pl.Datetime:
        df_feat = df_feat.with_columns(
            pl.col(timestamp_col).cast(pl.Datetime).alias("timestamp")
        )
    else:
        df_feat = df_feat.with_columns(
            pl.col(timestamp_col).alias("timestamp")
        )

    # Extract all temporal features
    df_feat = df_feat.with_columns([
        pl.col("timestamp").dt.year().alias("year"),
        pl.col("timestamp").dt.month().alias("month"),
        pl.col("timestamp").dt.day().alias("day"),
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.weekday().alias("dayofweek"),
        pl.col("timestamp").dt.ordinal_day().alias("dayofyear"),
    ])

    # Add weekend indicator
    df_feat = df_feat.with_columns(
        is_weekend=pl.when(pl.col("dayofweek").is_in([5, 6])).then(1).otherwise(0)
    )

    # Lag features (by zone to maintain independence)
    df_feat = df_feat.with_columns([
        pl.col("value").shift(1).over("zone", "year", "month", "day").alias("value_lag_1h"),
        pl.col("value").shift(24).over("zone", "year", "month").alias("value_lag_24h"),
        pl.col("value").shift(168).over("zone").alias("value_lag_168h"),  # 1 week
    ])

    # Rolling averages (by zone)
    df_feat = df_feat.with_columns([
        pl.col("value").rolling_mean(window_size=24).over("zone").alias("value_rolling_24h"),
        pl.col("value").rolling_mean(window_size=168).over("zone").alias("value_rolling_168h"),
    ])

    # Hour-of-day statistics (by zone)
    hourly_stats = df_feat.group_by("zone", "hour").agg([
        pl.col("value").mean().alias("hour_avg"),
        pl.col("value").std().alias("hour_std"),
    ])
    df_feat = df_feat.join(hourly_stats, on=["zone", "hour"], how="left")

    # Month-of-year statistics (by zone)
    monthly_stats = df_feat.group_by("zone", "month").agg([
        pl.col("value").mean().alias("month_avg"),
        pl.col("value").std().alias("month_std"),
    ])
    df_feat = df_feat.join(monthly_stats, on=["zone", "month"], how="left")

    # Day-of-week statistics (by zone)
    dow_stats = df_feat.group_by("zone", "dayofweek").agg([
        pl.col("value").mean().alias("dow_avg"),
        pl.col("value").std().alias("dow_std"),
    ])
    df_feat = df_feat.join(dow_stats, on=["zone", "dayofweek"], how="left")

    # Drop the original timestamp column if it's not the standard name
    if timestamp_col != "timestamp":
        df_feat = df_feat.drop(timestamp_col)

    return df_feat

def train_zone_model(df_zone, zone_name, test_size=0.2):
    """
    Train XGBoost model for a specific zone
    """
    print(f"\nTraining model for {zone_name}...")

    # Feature columns
    feature_cols = [
        'hour', 'dayofweek', 'is_weekend', 'month', 'day',
        'value_lag_1h', 'value_lag_24h', 'value_lag_168h',
        'value_rolling_24h', 'value_rolling_168h',
        'hour_avg', 'hour_std',
        'month_avg', 'month_std',
        'dow_avg', 'dow_std'
    ]

    # Filter available columns
    feature_cols = [col for col in feature_cols if col in df_zone.columns]

    # Convert to numpy
    X = df_zone.select(feature_cols).to_numpy()
    y = df_zone.select("value").to_numpy().flatten()

    # Normalize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Train-test split (maintain time order)
    split_idx = int(len(X_scaled) * (1 - test_size))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

    # Train model
    model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluate
    y_pred_test = model.predict(X_test)
    y_pred_test_original = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_test_original))
    mae = mean_absolute_error(y_test_original, y_pred_test_original)
    r2 = r2_score(y_test_original, y_pred_test_original)

    # Calculate MAPE safely (avoid division by zero)
    # Only calculate MAPE for non-zero actual values
    mask = y_test_original != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_test_original[mask] - y_pred_test_original[mask]) / y_test_original[mask])) * 100
    else:
        mape = np.nan  # If all values are zero, MAPE is undefined
        print(f"  Warning: All test values are zero, MAPE is undefined")

    print(f"  RMSE: {rmse:.2f} MW")
    print(f"  MAE: {mae:.2f} MW")
    print(f"  R² Score: {r2:.4f}")
    if not np.isnan(mape):
        print(f"  MAPE: {mape:.2f}%")
    else:
        print(f"  MAPE: N/A (all test values are zero)")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n  Top 5 Features for {zone_name}:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"    - {row['feature']}: {row['importance']:.4f}")

    return {
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_cols': feature_cols,
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape if not np.isnan(mape) else None
        },
        'feature_importance': feature_importance
    }

def predict_month(zone, month, year_to_predict, df_input, model_dict, percentile_data=None):
    """
    Predict all hours for a specific month and zone

    Parameters:
    - zone: Zone name (e.g., "Connecticut")
    - month: Month number (1-12)
    - year_to_predict: Year to predict
    - df_input: Input dataframe with historical data
    - model_dict: Dictionary containing model, scalers, and feature columns
    - percentile_data: Optional percentile data for bounds
    """

    model = model_dict['model']
    scaler_X = model_dict['scaler_X']
    scaler_y = model_dict['scaler_y']
    feature_cols = model_dict['feature_cols']

    # Get historical data for that month and zone
    historical = df_input.filter(
        (pl.col("zone") == zone) &
        (pl.col("month") == month)
    )

    if historical.shape[0] == 0:
        print(f"Warning: No historical data for {zone} in month {month}")
        return None

    predictions = []

    # Get days in month
    days_in_month = calendar.monthrange(year_to_predict, month)[1]

    for day in range(1, days_in_month + 1):
        for hour in range(24):
            try:
                # Get historical data for this hour
                hour_data = historical.filter(pl.col("hour") == hour)

                # Calculate day of week for prediction date
                pred_date = datetime(year_to_predict, month, day)
                dow = pred_date.weekday()
                is_weekend = 1 if dow in [5, 6] else 0

                # Create feature vector
                features_dict = {
                    'hour': float(hour),
                    'dayofweek': float(dow),
                    'is_weekend': float(is_weekend),
                    'month': float(month),
                    'day': float(day),
                    'value_lag_1h': float(
                        hour_data.select("value_lag_1h").mean().item() or historical.select("value").mean().item()),
                    'value_lag_24h': float(
                        hour_data.select("value_lag_24h").mean().item() or historical.select("value").mean().item()),
                    'value_lag_168h': float(
                        hour_data.select("value_lag_168h").mean().item() or historical.select("value").mean().item()),
                    'value_rolling_24h': float(hour_data.select("value_rolling_24h").mean().item() or historical.select(
                        "value").mean().item()),
                    'value_rolling_168h': float(
                        hour_data.select("value_rolling_168h").mean().item() or historical.select(
                            "value").mean().item()),
                    'hour_avg': float(
                        hour_data.select("hour_avg").mean().item() or historical.select("value").mean().item()),
                    'hour_std': float(hour_data.select("hour_std").mean().item() or 0),
                    'month_avg': float(historical.select("month_avg").mean().item()),
                    'month_std': float(historical.select("month_std").mean().item()),
                    'dow_avg': float(
                        hour_data.select("dow_avg").mean().item() or historical.select("value").mean().item()),
                    'dow_std': float(hour_data.select("dow_std").mean().item() or 0),
                }

                # Build feature array in correct order
                features = np.array([[features_dict.get(col, 0.0) for col in feature_cols]])

                # Scale and predict
                features_scaled = scaler_X.transform(features)
                pred_scaled = model.predict(features_scaled)[0]
                pred_original = scaler_y.inverse_transform([[pred_scaled]])[0][0]

                # Get percentile bounds from historical data
                hist_values = hour_data.select("value").to_numpy().flatten()

                if len(hist_values) > 0:
                    p10 = float(np.percentile(hist_values, 10))
                    p25 = float(np.percentile(hist_values, 25))
                    p75 = float(np.percentile(hist_values, 75))
                    p90 = float(np.percentile(hist_values, 90))
                    hist_min = float(np.min(hist_values))
                    hist_max = float(np.max(hist_values))
                    hist_avg = float(np.mean(hist_values))
                else:
                    p10 = p25 = p75 = p90 = hist_min = hist_max = hist_avg = float(pred_original)

                predictions.append({
                    'timestamp': pred_date.replace(hour=hour),
                    'zone': zone,
                    'hour': hour,
                    'day': day,
                    'predicted_mw': float(pred_original),
                    'p10': p10,
                    'p25': p25,
                    'p75': p75,
                    'p90': p90,
                    'historical_min': hist_min,
                    'historical_max': hist_max,
                    'historical_avg': hist_avg,
                })
            except Exception as e:
                # Silently skip errors for now
                continue

    if len(predictions) == 0:
        print(f"Warning: No predictions generated for {zone} in month {month}")
        return None

    return pl.DataFrame(predictions)