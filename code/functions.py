import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import warnings

SHIFT_EPSILON = 1e-6 # Use a small epsilon to avoid issues with exact zero

def holt_winters_imputation_and_expand(
    time_series,
    seasonal_periods,
    interim_fill_method='linear',
    left_expand=0,
    right_expand=0,
    use_boxcox_if_possible=True # Control whether to attempt Box-Cox
):
    """
    Impute missing values and optionally expand the series using multiple Holt-Winters models.
    Handles potential negative values by shifting data before applying Box-Cox if enabled.

    Parameters:
    - time_series: pd.Series
        The time series data with NaN values to be imputed. Must have a DatetimeIndex
        or a numerical index with a discernible frequency.
    - seasonal_periods: int
        The number of observations per cycle of seasonality. Must be >= 2.
    - interim_fill_method: str, optional (default='linear')
        Method to temporarily fill missing values for model fitting
        (e.g., 'linear', 'time', 'nearest', 'spline', 'polynomial').
    - left_expand: int, optional (default=0)
        Number of periods to expand backward in time (backcast).
    - right_expand: int, optional (default=0)
        Number of periods to expand forward in time (forecast).
    - use_boxcox_if_possible: bool, optional (default=True)
        If True, attempt to use Box-Cox transformation. If data contains non-positive
        values, it will be shifted to become positive before Box-Cox is applied.
        If False, Box-Cox will not be used (`use_boxcox=False`).

    Returns:
    - expanded_series: pd.Series
        The series with imputed values and optionally expanded data on the original scale.

    Raises:
    - ValueError: If input is not a pandas Series, if seasonal_periods < 2,
                  if the series has too few non-NaN values, or if frequency cannot be inferred.
    - ConvergenceWarning: May be raised by statsmodels if optimization fails.
    """
    # Input Validation
    if not isinstance(time_series, pd.Series):
        raise ValueError("Input time_series must be a pandas Series.")
    if not isinstance(seasonal_periods, int) or seasonal_periods < 2:
         raise ValueError("seasonal_periods must be an integer >= 2.")
    if time_series.notna().sum() < 2 * seasonal_periods:
        warnings.warn(f"Series has less than 2 * seasonal_periods ({2*seasonal_periods}) non-NaN values. "
                      "Model fitting might be unstable or fail.", UserWarning)
    if time_series.notna().sum() < 2:
         raise ValueError("Series must have at least 2 non-NaN values to fit models.")

    original_series = time_series.copy()
    series_for_fitting = original_series.copy()

    # Step 1: Temporarily fill missing values for model fitting
    # Check for sufficient non-NaN values before interpolation
    if series_for_fitting.isna().all():
        raise ValueError("Cannot interpolate an all-NaN series.")
    if series_for_fitting.notna().sum() == 1:
         # Handle cases where only one value exists - maybe fill with that value?
         # Or raise error as interpolation might not be well-defined.
         # For now, let interpolate handle it, but be aware.
         warnings.warn("Series has only one non-NaN value; interpolation might be trivial.", UserWarning)

    # Choose interpolation order if applicable (for spline/polynomial)
    interpolation_order = 3
    if interim_fill_method in ['spline', 'polynomial']:
        order_candidate = min(interpolation_order, series_for_fitting.notna().sum() - 1)
        if order_candidate < 1 :
             warnings.warn(f"Cannot use {interim_fill_method} with order {interpolation_order} "
                           f"due to insufficient ({series_for_fitting.notna().sum()}) non-NaN points. "
                           "Falling back to linear.", UserWarning)
             interim_fill_method = 'linear' # Fallback
             interim_filled = series_for_fitting.interpolate(method=interim_fill_method)
        else:
             interim_filled = series_for_fitting.interpolate(method=interim_fill_method, order=order_candidate)
    else:
         interim_filled = series_for_fitting.interpolate(method=interim_fill_method)

    # --- Data Transformation (Shift for Box-Cox) ---
    shift_value = 0.0
    actual_use_boxcox = False # Flag to track if boxcox is actually used

    if use_boxcox_if_possible:
        min_val = interim_filled.min()
        if min_val <= 0:
            shift_value = abs(min_val) + SHIFT_EPSILON
            series_to_fit = interim_filled + shift_value
            print(f"Data contains non-positive values. Shifting data by {shift_value:.4f} before applying Box-Cox.")
            actual_use_boxcox = True # We are shifting, so Box-Cox will be attempted
        elif np.all(interim_filled > 0):
             series_to_fit = interim_filled
             actual_use_boxcox = True # Data is already positive
             print("Data is positive. Proceeding with Box-Cox.")
        else: # Should not happen if min_val > 0, but as safety
             series_to_fit = interim_filled
             actual_use_boxcox = False # Cannot use Box-Cox if conditions aren't met (though covered above)
             print("Data conditions not suitable for Box-Cox after checks; Box-Cox disabled.")

    else:
        series_to_fit = interim_filled
        actual_use_boxcox = False
        print("Box-Cox explicitly disabled.")
    # --- End Transformation ---

    # Step 2: Fit 4 Holt-Winters models
    models = []
    configs = [
        {"trend": "add", "seasonal": "add", "damped_trend": False},
        {"trend": "add", "seasonal": "mul", "damped_trend": False},
        {"trend": "add", "seasonal": "add", "damped_trend": True},
        {"trend": "add", "seasonal": "mul", "damped_trend": True},
    ]

    fitted_models = []
    for config in configs:
        try:
            # Ensure seasonal='mul' is only used if data is positive AFTER potential shift
            current_seasonal = config['seasonal']
            if current_seasonal == 'mul' and series_to_fit.min() <= 0:
                warnings.warn(f"Cannot use seasonal='mul' with non-positive data (min={series_to_fit.min():.4f}) "
                              f"even after shifting for Box-Cox. Skipping this model configuration: {config}", UserWarning)
                continue # Skip this model configuration

            model = ExponentialSmoothing(
                series_to_fit, # Use the potentially shifted series
                seasonal_periods=seasonal_periods,
                trend=config["trend"],
                seasonal=current_seasonal,
                damped_trend=config["damped_trend"],
                use_boxcox=actual_use_boxcox, # Use determined flag
                initialization_method="estimated"
            )
            # Filter convergence warnings specifically if desired
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning) # Treat runtime warnings as errors initially
                warnings.filterwarnings("error", category=UserWarning) # Treat user warnings as errors initially
                warnings.filterwarnings("ignore", message="Optimization failed to converge.") # Ignore specific convergence warning message
                try:
                    fitted_models.append(model.fit())
                except (RuntimeWarning, UserWarning) as w:
                     # Catch other warnings treated as errors
                     print(f"Warning/Error during model fitting for config {config}: {w}")
                     # Optionally decide not to add the model or handle differently
                except Exception as e:
                    print(f"Error fitting model for config {config}: {e}")


        except ValueError as e:
            print(f"ValueError fitting model for config {config}: {e}. Skipping this model.")
        except Exception as e: # Catch other potential errors during model setup
             print(f"Unexpected error setting up model for config {config}: {e}. Skipping this model.")


    if not fitted_models:
        raise RuntimeError("No Holt-Winters models could be successfully fitted.")

    # Step 3: Calculate mean predictions (on the transformed scale if shifted)
    # Use np.nanmean to handle cases where some models might have failed/produced NaNs
    fitted_values_transformed = np.nanmean([m.fittedvalues for m in fitted_models], axis=0)

    # Step 4: Inverse Transform Fitted Values
    fitted_values_original_scale = fitted_values_transformed - shift_value

    # Step 5: Impute missing values in the *original* series using *inverse-transformed* fitted values
    imputed_series = original_series.copy()
    nan_mask = original_series.isna()
    # Ensure indices align - create a temporary series from fitted values
    fitted_series_original_scale = pd.Series(fitted_values_original_scale, index=series_to_fit.index)
    # Impute using the aligned fitted values
    imputed_series[nan_mask] = fitted_series_original_scale[nan_mask]


    # Step 6: Expand the series if needed
    expanded_series = imputed_series.copy() # Start with the imputed series

    if left_expand > 0 or right_expand > 0:
        # --- Get Frequency ---
        freq = pd.infer_freq(original_series.index)
        if freq is None and isinstance(original_series.index, pd.DatetimeIndex):
             # Attempt to calculate from differences if DatetimeIndex
             diffs = np.diff(original_series.index)
             if len(diffs) > 0:
                 freq = pd.tseries.frequencies.to_offset(np.min(diffs))
                 if freq:
                      warnings.warn(f"Frequency not explicitly set, inferred as '{freq.rule_code}' from index differences.", UserWarning)
                 else:
                      raise ValueError("Could not infer frequency from DatetimeIndex differences for expansion.")
             else:
                 raise ValueError("Cannot infer frequency for expansion from a series with < 2 points.")
        elif freq is None :
             # Check if index is integer-based range-like
             if isinstance(original_series.index, pd.RangeIndex) or (np.all(np.diff(original_series.index) == 1)):
                 freq = 1 # Assume unit frequency for basic numeric index
                 warnings.warn("Frequency not explicitly set, assuming unit frequency (1) for numeric index.", UserWarning)
             else:
                raise ValueError("Could not infer frequency for expansion. Ensure series index has a frequency (e.g., DatetimeIndex) or is a simple range.")


        # --- Forecast ---
        if right_expand > 0:
            forecasts_transformed = np.nanmean([m.forecast(steps=right_expand) for m in fitted_models], axis=0)
            forecasts_original_scale = forecasts_transformed - shift_value

            # Create forecast index
            last_index_val = original_series.index[-1]
            if isinstance(original_series.index, pd.DatetimeIndex):
                 forecast_index = pd.date_range(start=last_index_val + freq, periods=right_expand, freq=freq)
            elif isinstance(freq, int): # Handle numeric index
                 forecast_index = pd.RangeIndex(start=last_index_val + freq, stop=last_index_val + freq * (right_expand + 1), step=freq)
            else:
                 raise TypeError(f"Unsupported index type or frequency for forecast index generation: {type(original_series.index)}, freq={freq}")

            forecast_series = pd.Series(forecasts_original_scale, index=forecast_index)
            expanded_series = pd.concat([expanded_series, forecast_series])


        # --- Backcast ---
        if left_expand > 0:
            # Predict method needs integer indices relative to the start of the fitted data
            start_pred_idx = len(series_to_fit) # Prediction starts *after* the fitted data ends
            end_pred_idx = len(series_to_fit) + left_expand -1 # Predict 'left_expand' steps into the past conceptually

            # NOTE: statsmodels .predict() for HW doesn't directly support reliable backcasting
            # in the sense of predicting values *before* the start of the training data.
            # The `start` and `end` parameters are typically used for in-sample or future forecasts.
            # Attempting backcasting this way might be unreliable.
            # A more robust approach might involve reversing the series, fitting, forecasting, and reversing back.
            # However, sticking to the original code's apparent intent for now:
            # We will simulate backcasting by forecasting on a *reversed* series if needed,
            # or acknowledge the limitation.

            # Simple approach (potentially less accurate for backcasting): Use predict relative to end
            # This predict `-left_expand` tries to get values from the model's state *before* start.
            try:
                 # Use predict method, indices relative to the start of the *fitted* series
                 backcasts_transformed = np.nanmean([m.predict(start=-left_expand, end=-1) for m in fitted_models], axis=0)
                 if len(backcasts_transformed) != left_expand:
                      warnings.warn(f"Backcasting using predict returned {len(backcasts_transformed)} points, expected {left_expand}. May be unreliable.", UserWarning)
                      # Pad with NaN if needed, though this indicates predict didn't work as hoped
                      if len(backcasts_transformed) < left_expand:
                           backcasts_transformed = np.pad(backcasts_transformed, (left_expand - len(backcasts_transformed), 0), constant_values=np.nan)

                 backcasts_original_scale = backcasts_transformed - shift_value

                 # Create backcast index
                 first_index_val = original_series.index[0]
                 if isinstance(original_series.index, pd.DatetimeIndex):
                     backcast_index = pd.date_range(end=first_index_val - freq, periods=left_expand, freq=freq)
                 elif isinstance(freq, int): # Handle numeric index
                     backcast_index = pd.RangeIndex(start=first_index_val - freq*left_expand, stop=first_index_val, step=freq)
                 else:
                     raise TypeError(f"Unsupported index type or frequency for backcast index generation: {type(original_series.index)}, freq={freq}")

                 backcast_series = pd.Series(backcasts_original_scale, index=backcast_index)
                 expanded_series = pd.concat([backcast_series, expanded_series]) # Prepend backcasts

            except Exception as e:
                 warnings.warn(f"Could not perform backcasting using predict method: {e}. Left expansion skipped.", UserWarning)


    # Ensure the final series is sorted by index if concatenation happened
    expanded_series = expanded_series.sort_index()

    return expanded_series


def plot_imputation_results(original_series, imputed_series, title="Holt-Winters Imputation Results"):
    """
    Plot the original time series with missing values and the imputed time series.

    Parameters:
    - original_series: pd.Series
        The original time series with NaN values.
    - imputed_series: pd.Series
        The time series after imputation.
    - title: str
        Title for the plot.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot the imputed series
    plt.plot(imputed_series.index, imputed_series, label="Imputed & Expanded Series", 
             linestyle="-", marker="o", color="green")
    
    # Plot the original series
    plt.plot(original_series.index, original_series, label="Original Series (Observed)", 
             linestyle="--", marker="o", alpha=0.7, color="blue")
    
    # Highlight missing values
    missing_indices = original_series[original_series.isna()].index
    plt.scatter(missing_indices, imputed_series[missing_indices], color="red", 
                label="Imputed Points", zorder=5)

    # Customization
    plt.title(title, fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()



