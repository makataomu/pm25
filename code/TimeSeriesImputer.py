import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import kpss
from scipy.special import boxcox1p, inv_boxcox1p
import itertools
import warnings

class TimeSeriesPreprocessor:
    def __init__(self, seasonal_period=365):
        self.seasonal_period = seasonal_period
        self.trend = None
        self.seasonality = None
        self.lambda_param = None
        self.mean = None
        self.std = None
        self.lambdas = None
        self.original_first_value = None
        
    def kpss_test(self, series):
        try:
            statistic, p_value, _, _ = kpss(series, regression='c')
            return p_value < 0.05  # Return True if the series is not stationary
        except Exception as e:
            warnings.warn(f"KPSS test failed: {e}")
            return False
        
    def determine_seasonal_differencing(self, series, fs_threshold=0.64):
        try:
            decomposition = seasonal_decompose(series, period=self.seasonal_period)
            seasonal_strength = 1 - (np.var(decomposition.resid) / np.var(decomposition.seasonal + decomposition.resid))
            return 1 if seasonal_strength >= fs_threshold else 0
        except Exception as e:
            warnings.warn(f"Failed to determine seasonal differencing: {e}")
            return 0

    def remove_trend(self, series, method='diff'):
        try:
            if method == 'decomposition':
                decomposition = seasonal_decompose(series, period=self.seasonal_period)
                self.trend = decomposition.trend
                return series - decomposition.trend
            elif method == 'diff':
                self.original_first_value = series[0]
                return np.diff(series, prepend=series[0])
        except Exception as e:
            warnings.warn(f"Failed to remove trend: {e}")
            return series

    def remove_seasonality(self, series):
        try:
            decomposition = seasonal_decompose(series, period=self.seasonal_period)
            self.seasonality = decomposition.seasonal
            return series - decomposition.seasonal
        except Exception as e:
            warnings.warn(f"Failed to remove seasonality: {e}")
            return series

    def apply_boxcox(self, series, lambda_param=None):
        try:
            if lambda_param is None:
                self.lambda_param = stats.boxcox_normmax(series + 1)
            else:
                self.lambda_param = lambda_param
            return boxcox1p(series, self.lambda_param), self.lambda_param
        except Exception as e:
            warnings.warn(f"Failed to apply Box-Cox transformation: {e}")
            return series, -1

    def apply_log(self, series):
        try:
            if (series <= 0).any():
                warnings.warn("Log transformation requires all values to be positive. Skipping log transformation.")
                return series
            return np.log(series)
        except Exception as e:
            warnings.warn(f"Failed to apply log transformation: {e}")
            return series

    def standardize(self, series):
        try:
            self.mean = series.mean()
            self.std = series.std()
            return (series - self.mean) / self.std
        except Exception as e:
            warnings.warn(f"Failed to standardize series: {e}")
            return series

    def fit_transform(self, series, remove_trend=True, remove_seasonality=True, 
                     apply_boxcox=False, apply_log=False, standardize=False, trend_method='diff'):
        transformed = series.copy()
        self.transforms = []
        
        if remove_trend:
            transformed = self.remove_trend(transformed, method=trend_method)
            self.transforms.append('trend')
            
        if remove_seasonality:
            seasonal_diffs = self.determine_seasonal_differencing(transformed)
            if seasonal_diffs > 0:
                transformed = np.diff(transformed, n=seasonal_diffs)
            else:
                transformed = self.remove_seasonality(transformed)
            self.transforms.append('seasonality')
            
        if apply_boxcox:
            transformed, _ = self.apply_boxcox(transformed)
            self.transforms.append('boxcox')
            
        if apply_log:
            transformed = self.apply_log(transformed)
            self.transforms.append('log')

        if standardize:
            transformed = self.standardize(transformed)
            self.transforms.append('standardize')
            
        return transformed

    def inverse_transform_predictions(self, predictions):
        transformed = predictions.copy()

        for trans in self.transforms[::-1]:
            if 'standardize' == trans:
                transformed = (transformed * self.std) + self.mean

            if 'log' == trans:
                transformed = np.exp(transformed)

            if 'boxcox' == trans:
                if self.lambda_param != -1:
                    transformed = inv_boxcox1p(transformed, self.lambda_param)

            if 'seasonality' == trans:
                start_idx = len(self.trend) % self.seasonal_period if self.trend is not None else 0
                seasonal_indices = np.arange(start_idx, start_idx + len(predictions)) % self.seasonal_period
                seasonal_pattern = self.seasonality[seasonal_indices]
                transformed = transformed + seasonal_pattern

            if 'trend' == trans:
                if hasattr(self, 'original_first_value'):
                    reversed_series = np.cumsum(transformed)
                    transformed = reversed_series + self.original_first_value
                else:
                    warnings.warn("Trend removal used differencing, but original_first_value is not set. Skipping trend reversal.")
        
        return transformed

    def create_pipeline(self, series, steps):
        """
        steps: list of transformations ['trend', 'seasonality', 'boxcox', 'log', 'standardize']
        Returns: dict with all possible combinations of transformations
        """
        results = {'original': series.copy()}

        self.lambdas = {}
        
        for r in range(1, len(steps) + 1):
            for combo in itertools.combinations(steps, r):
                transformed = series.copy()
                name = []
                
                for step in combo:
                    if step == 'trend':
                        transformed = self.remove_trend(transformed)
                        name.append('trend')
                    elif step == 'seasonality':
                        transformed = self.remove_seasonality(transformed)
                        name.append('seasonality')
                    elif step == 'boxcox':
                        transformed, lambda_param = self.apply_boxcox(transformed)
                        self.lambdas["_".join(combo)] = lambda_param

                        name.append('boxcox')
                    elif step == 'log':
                        transformed = self.apply_log(transformed)
                        name.append('log')
                    elif step == 'standardize':
                        transformed = self.standardize(transformed)
                        name.append('standardize')
                    
                    transformed = transformed[~np.isnan(transformed)]

                transformed = transformed[~np.isnan(transformed)]
                results['_'.join(name)] = transformed
                
        return results

# usage
# data = df['value'].values.copy()
# preprocessor = TimeSeriesPreprocessor(seasonal_period=365)

# steps = ['log', 'trend', 'seasonality', 'boxcox', 'standardize']
# transformed_series = preprocessor.create_pipeline(data, steps)
# 3.0 modeling statsforecast notebook