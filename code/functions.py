import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt


def holt_winters_imputation_and_expand(
    time_series,
    seasonal_periods,
    interim_fill_method='linear',
    left_expand=0,
    right_expand=0
):
    """
    Impute missing values and optionally expand the series using multiple Holt-Winters models.

    Parameters:
    - time_series: pd.Series
        The time series data with NaN values to be imputed.
    - seasonal_periods: int
        The number of observations per cycle of seasonality.
    - interim_fill_method: str, optional
        Method to temporarily fill missing values.
    - left_expand: int, optional
        Number of periods to expand backward in time.
    - right_expand: int, optional
        Number of periods to expand forward in time.

    Returns:
    - expanded_series: pd.Series
        The series with imputed values and optionally expanded data.
    """
    # Check if the input is a pandas Series
    if not isinstance(time_series, pd.Series):
        raise ValueError("Input time_series must be a pandas Series.")

    # Step 1: Temporarily fill missing values
    interim_filled = time_series.interpolate(method=interim_fill_method)

    # Step 2: Fit 4 Holt-Winters models with different configurations
    models = []
    models.append(ExponentialSmoothing(
        interim_filled,
        seasonal_periods=seasonal_periods,
        trend="add",
        seasonal="add",
        use_boxcox=True,
        initialization_method="estimated"
    ).fit())
    models.append(ExponentialSmoothing(
        interim_filled,
        seasonal_periods=seasonal_periods,
        trend="add",
        seasonal="mul",
        use_boxcox=True,
        initialization_method="estimated"
    ).fit())
    models.append(ExponentialSmoothing(
        interim_filled,
        seasonal_periods=seasonal_periods,
        trend="add",
        seasonal="add",
        damped_trend=True,
        use_boxcox=True,
        initialization_method="estimated"
    ).fit())
    models.append(ExponentialSmoothing(
        interim_filled,
        seasonal_periods=seasonal_periods,
        trend="add",
        seasonal="mul",
        damped_trend=True,
        use_boxcox=True,
        initialization_method="estimated"
    ).fit())

    # Step 3: Calculate mean predictions across all models
    fitted_values = np.mean([model.fittedvalues for model in models], axis=0)

    # Step 4: Impute missing values
    imputed_series = time_series.copy()
    imputed_series[time_series.isna()] = fitted_values[time_series.isna()]

    # Step 5: Expand the series if needed
    if left_expand > 0 or right_expand > 0:
        # Forecast future values
        forecasts = np.mean([model.forecast(steps=right_expand) for model in models], axis=0) if right_expand > 0 else []
        # Backcast past values
        backcasts = np.mean([model.predict(start=-left_expand, end=-1) for model in models], axis=0) if left_expand > 0 else []

        # Create indices for backcasts and forecasts
        backcast_index = range(imputed_series.index[0] - left_expand, imputed_series.index[0]) if left_expand > 0 else []
        forecast_index = range(imputed_series.index[-1] + 1, imputed_series.index[-1] + right_expand + 1) if right_expand > 0 else []

        # Convert backcasts and forecasts to series
        backcast_series = pd.Series(backcasts, index=backcast_index) if left_expand > 0 else pd.Series()
        forecast_series = pd.Series(forecasts, index=forecast_index) if right_expand > 0 else pd.Series()

        # Concatenate the series
        expanded_series = pd.concat([backcast_series, imputed_series, forecast_series])
    else:
        expanded_series = imputed_series

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



