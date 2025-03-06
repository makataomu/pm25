from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlforecast import MLForecast
from mlforecast.target_transforms import (
    Differences, AutoDifferences, AutoSeasonalDifferences, AutoSeasonalityAndDifferences,
    LocalStandardScaler, LocalMinMaxScaler, LocalBoxCox
)
from window_ops.expanding import expanding_mean
from window_ops.rolling import rolling_mean
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, Lasso, SGDRegressor
from itertools import combinations, chain
import pickle

def determine_max_lags(train_df, min_lags=10, max_fraction=0.5, max_limit=400):
    """ Determines the maximum number of lags based on train set size. """
    max_lags = min(int(len(train_df) * max_fraction), max_limit)
    return max(max_lags, min_lags)  # Ensure at least min_lags

def determine_dynamic_max_lags(train_df, min_lags=15, max_fraction=0.5, max_limit=400):
    """
    Dynamically determines a range of max_lags values based on train size.
    Returns a list of `max_lags` values to be tested.
    """
    base_max_lags = min(int(len(train_df) * max_fraction), max_limit)  # Base max lags
    
    # Create diverse lag options (quarter, half, full, and extended)
    max_lags_list = [
        # max(min_lags, base_max_lags // 6),   # Small max_lags
        max(min_lags, base_max_lags // 4),   # Small max_lags
        max(min_lags, base_max_lags // 2),   # Medium max_lags
        base_max_lags,                      # Full max_lags
    ]
    
    # Remove duplicates and ensure sorted order
    max_lags_list = sorted(set(max_lags_list))

    return max_lags_list


def generate_lagged_features(train_df, target_col, max_lags):
    """Generates lagged features while keeping the `ds` column."""
    # lagged_features = pd.concat([
        # train_df[[target_col, "ds"]].assign(**{f'lag_{lag}': train_df[target_col].shift(lag)}) for lag in range(1, max_lags + 1)
    # ], axis=1)
    # train_df.set_index("ds", inplace=True)
    lagged_features = pd.concat([
        train_df[target_col].shift(lag).rename(f'lag_{lag}') for lag in range(1, max_lags + 1)
    ], axis=1)

    # Drop missing values (due to shifting) and reset index
    lagged_features = lagged_features.dropna().reset_index(drop=True)
    
    return lagged_features


def select_important_lags(train_df, target_col, max_lags, model=RandomForestRegressor(), num_of_lags=10):
    """ Selects the most important lags based on feature importance analysis. """
    lagged_features = generate_lagged_features(train_df, target_col, max_lags)
    y = train_df[target_col][max_lags:]  # Align target values
    
    if lagged_features.shape[0] != len(y):  # Avoid mismatched sizes
        lagged_features = lagged_features.iloc[:len(y)]
    
    model.fit(lagged_features, y)
    feature_importances = model.feature_importances_
    important_lags = [i + 1 for i in np.argsort(feature_importances)[-num_of_lags:]]  # Select top lags
    
    return sorted(important_lags)

def select_important_lags_extended(train_df, target_col, max_lags, model=RandomForestRegressor(), num_of_lags_list=[5, 10, 15]):
    """ Selects the most important lags based on feature importance analysis for multiple numbers of lags."""
    lagged_features = generate_lagged_features(train_df, target_col, max_lags)

    y = train_df[target_col][max_lags:]
    if lagged_features.shape[0] != len(y):  # Avoid mismatched sizes
        lagged_features = lagged_features.iloc[:len(y)]
    
    model.fit(lagged_features, y)
    feature_importances = model.feature_importances_
    
    important_lags_lists = {}
    for num_of_lags in num_of_lags_list:
        important_lags = [i + 1 for i in np.argsort(feature_importances)[-num_of_lags:]]  # Select top lags
        name = f"lags_{max_lags}_features_{num_of_lags}"  # Generate a meaningful name
        important_lags_lists[name] = [int(x) for x in sorted(important_lags)]  # Store with name
    
    return important_lags_lists

def get_optimal_lags(train_df, target_col, model=RandomForestRegressor(), ratios=[0.33, 0.66, 1]):
    """ Selects the most important lags dynamically based on train size. """
    max_lags_list = determine_dynamic_max_lags(train_df)  # Get dynamic max_lags
    results = {}

    for max_lags in max_lags_list:
        num_of_lags_list = [int(max_lags * ratio) for ratio in ratios]  # Various % of max_lags

        # Select important lags and store them with meaningful names
        selected_lags = select_important_lags_extended(train_df, target_col, max_lags, model, num_of_lags_list)
        
        # Merge into results dictionary
        results.update(selected_lags)

    return results

def mape_met(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

# Function to dynamically determine seasonal and differencing parameters
def get_dynamic_transforms(train_df):
    max_diffs = min(len(train_df) // 2, 380)  # Avoid excessive differencing
    season_length = min(len(train_df) // 3, 365)  # Estimate reasonable seasonality

    target_transforms = [
        AutoDifferences(max_diffs=max_diffs), 
        AutoSeasonalDifferences(season_length=season_length, max_diffs=max_diffs), 
        AutoSeasonalityAndDifferences(max_season_length=season_length, max_diffs=max_diffs),
        LocalStandardScaler(), 
        LocalMinMaxScaler(), 
        LocalBoxCox()
    ]
    return target_transforms

# Function to dynamically determine max lags
def determine_max_lags(train_df, min_lags=20, max_fraction=0.5, max_limit=400):
    """ Determines the maximum number of lags based on train set size. """
    max_lags = min(int(len(train_df) * max_fraction), max_limit)
    return max(max_lags, min_lags)


# Function to validate transform combinations
def filter_conflicting_transforms(transform_combination):
    conflicting_transforms = {Differences, AutoDifferences, AutoSeasonalDifferences, AutoSeasonalityAndDifferences}
    scaler_transforms = {LocalStandardScaler, LocalMinMaxScaler, LocalBoxCox}
    
    if sum(1 for t in transform_combination if type(t) in conflicting_transforms) > 1:
        return False
    if sum(1 for t in transform_combination if type(t) in scaler_transforms) > 1:
        return False
    return True

def return_sgdreg_name(model_name):
    if "SGDRegressor" in model_name:
        return "SGDRegressor"
    return model_name

# Model Evaluation Pipeline
def evaluate_models(train_df, test_df, models, target_transforms, lag_transforms_options, optimal_lags_list, date_features=['dayofweek', 'month']):
    """
    Evaluates multiple models with different transformations, lag selections, and lag transformations.
    Now accepts precomputed `optimal_lags_list` instead of calculating inside.
    """
    results = []

    # Validate transform combinations
    valid_transform_combinations = [()] + list(chain(combinations(target_transforms, 1), combinations(target_transforms, 2)))
    valid_transform_combinations = [tc for tc in valid_transform_combinations if filter_conflicting_transforms(tc)]

    max_test_length = len(test_df)  # Full test period

    # Define test segment lengths: 1-6 months, then 8, 10, 12, 16, 20, etc.
    test_lengths = list(range(30, 181, 30)) + [240, 300, 360, 480, 600, 720, max_test_length]  # Days-based segmentation

    # Filter lengths within available test period
    test_lengths = [t for t in test_lengths if t <= max_test_length]

    total_fits = len(models) * len(valid_transform_combinations) * len(optimal_lags_list) * len(lag_transforms_options)
    print(f"Total model fits to run: {total_fits}")

    fit_num = 0
    for lag_name, optimal_lags in optimal_lags_list.items():  # Now uses precomputed lags
        for transform_combination in valid_transform_combinations:
            for lag_transforms in lag_transforms_options:
                for model_name, model in models.items():
                    print(f"{fit_num}/{total_fits} Training {model_name} with transforms {transform_combination}, lags {optimal_lags}, and lag_transforms {lag_transforms}...")

                    try:
                        fcst = MLForecast(
                            models=[model],
                            freq='D',
                            lags=optimal_lags,
                            target_transforms=list(transform_combination),
                            date_features=date_features,
                            num_threads=1,
                            lag_transforms=lag_transforms,
                        )
                        
                        # Fit the model
                        fcst.fit(train_df)
                        
                        # Predict
                        predictions = fcst.predict(h=max_test_length)
                        test_df_copy = test_df.copy()
                        test_df_copy['forecast'] = predictions[model_name].values       

                        error_dict = {}

                        for test_length in test_lengths:
                            eval_subset = test_df_copy.iloc[:test_length]  # Take subset for evaluation
                            # print('eval_subset', eval_subset.shape, eval_subset)
                            # raise KeyError('pashol na')
                            # Store error in the dictionary
                            error_dict[f"test_{test_length}_days"] = mape_met(eval_subset['y'].values,  eval_subset['forecast'].values)

                        # Store results
                        # Merge predictions back to maintain the `ds` column
                        results.append({
                            "Model": model_name,
                            "Transforms": str(transform_combination),
                            "Lags": optimal_lags,
                            "Lag Transforms": str(lag_transforms),
                            "Lag Name": lag_name,
                            **error_dict  # Expand error dictionary into separate columns
                        })
                        print(f"{model_name} MAPE: {error_dict[f'test_{max_test_length}_days']:.2f}% with transforms {transform_combination}, lags {optimal_lags}, and lag_transforms {lag_transforms}")
                        
                    except Exception as e:
                        print(f"Skipping combination {fit_num} due to error: {e}")

                    fit_num += 1
    return pd.DataFrame(results)

import csv
def save_results_generator(results_generator, test_lengths, filename="forecast_results.csv"):
    """ Saves results from a generator directly to a CSV file """
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "Transforms", "Lags", "Lag Transforms", "Lag Name"] +
                                            [f"test_{t}_days" for t in test_lengths])
        writer.writeheader()
        for row in results_generator:
            writer.writerow(row)

def evaluate_models_generator(train_df, test_df, models, target_transforms, lag_transforms_options, optimal_lags_list, test_lengths, date_features=['dayofweek', 'month']):
    """
    Evaluates multiple models with different transformations, lag selections, and lag transformations.
    Now accepts precomputed `optimal_lags_list` instead of calculating inside.
    """

    # Validate transform combinations
    valid_transform_combinations = [()] + list(chain(combinations(target_transforms, 1), combinations(target_transforms, 2)))
    valid_transform_combinations = [tc for tc in valid_transform_combinations if filter_conflicting_transforms(tc)]

    max_test_length = len(test_df)  # Full test period

    # Define test segment lengths: 1-6 months, then 8, 10, 12, 16, 20, etc.
    # test_lengths = list(range(30, 181, 30)) + [240, 300, 360, 480, 600, 720, max_test_length]  # Days-based segmentation

    # Filter lengths within available test period
    # test_lengths = [t for t in test_lengths if t <= max_test_length]

    total_fits = len(models) * len(valid_transform_combinations) * len(optimal_lags_list) * len(lag_transforms_options)
    print(f"Total model fits to run: {total_fits}")

    fit_num = 0
    for lag_name, optimal_lags in optimal_lags_list.items():  # Now uses precomputed lags
        for transform_combination in valid_transform_combinations:
            for lag_transforms in lag_transforms_options:
                for model_name, model in models.items():
                    print(f"{fit_num}/{total_fits} Training {model_name} with transforms {transform_combination}, lags {optimal_lags}, and lag_transforms {lag_transforms}...")

                    try:
                        fcst = MLForecast(
                            models=[model],
                            freq='D',
                            lags=optimal_lags,
                            target_transforms=list(transform_combination),
                            date_features=date_features,
                            num_threads=1,
                            lag_transforms=lag_transforms,
                        )
                        
                        # Fit the model
                        fcst.fit(train_df)
                        
                        # Predict
                        predictions = fcst.predict(h=max_test_length)
                        test_df_copy = test_df.copy()
                        test_df_copy['forecast'] = predictions[model_name].values       

                        error_dict = {}

                        for test_length in test_lengths:
                            eval_subset = test_df_copy.iloc[:test_length]  # Take subset for evaluation
                            # Store error in the dictionary
                            error_dict[f"test_{test_length}_days"] = mape_met(eval_subset['y'].values,  eval_subset['forecast'].values)

                        # Store results
                        # Merge predictions back to maintain the `ds` column
                        yield {
                            "Model": model_name,
                            "Transforms": str(transform_combination),
                            "Lags": optimal_lags,
                            "Lag Transforms": str(lag_transforms),
                            "Lag Name": lag_name,
                            **error_dict  # Expand error dictionary into separate columns
                        }
                        print(f"{model_name} MAPE: {error_dict[f'test_{max_test_length}_days']:.2f}% with transforms {transform_combination}, lags {optimal_lags}, and lag_transforms {lag_transforms}")
                        
                    except Exception as e:
                        print(f"Skipping combination {fit_num} due to error: {e}")

                    fit_num += 1
    # return pd.DataFrame(results)

import json
import re
import pandas as pd

def stringify_transform(transforms):
    """
    Convert transformation(s) into a standardized string format including parameters.
    
    - Handles both **single** transformations and **lists** of transformations.
    - Extracts parameters **only if `scaler_` exists**, otherwise just takes the class name.
    """
    
    if not isinstance(transforms, list):  # If it's a single transformation, wrap it in a list
        transforms = [transforms]

    transform_strings = []
    
    for transform in transforms:
        class_name = transform.__class__.__name__  # Get the class name
        
        # Check if the transform has a `scaler_` attribute
        if hasattr(transform, 'scaler_'):
            actual_transform = transform.scaler_
            
            # Extract all attributes dynamically
            attr_strings = []
            for attr in dir(actual_transform):
                if (not attr.startswith("_")) \
                    and (not callable(getattr(actual_transform, attr, None))) \
                    and (attr not in ['tails_', 'diffs_']) \
                :
                    attr_value = getattr(actual_transform, attr, None)
                    attr_strings.append(f"{attr}={attr_value}")
            
            # Format class name + parameters
            attr_str = ", ".join(attr_strings) if attr_strings else "NoParams"
            transform_strings.append(f"{class_name}({attr_str})")
        
        else:
            # If no `scaler_`, just store the class name
            transform_strings.append(class_name + '()')
    
    return " | ".join(transform_strings)  # Join multiple transformations with " | "


def parse_transform(transform_str):
    """
    Convert string representation back into a list of transformation objects.
    - Handles **single** and **multiple** transformations.
    - Extracts parameters dynamically if present.
    """
    
    transform_list = transform_str.split(" | ")  # Split multiple transforms
    parsed_transforms = []
    
    for transform_item in transform_list:
        if "(" in transform_item:  # If parameters exist
            class_name, params_str = transform_item.split("(", 1)
            params_str = params_str.rstrip(")")
            
            # Extract parameters into a dictionary
            params = {}
            if params_str != "NoParams":
                for param in params_str.split(", "):
                    key, value = param.split("=")
                    try:
                        params[key] = eval(value)  # Convert to appropriate type (int, float, etc.)
                    except:
                        params[key] = value  # Keep as string if eval fails
            
            # Dynamically create the object
            if class_name in globals():
                parsed_transforms.append(globals()[class_name](**params))
            else:
                raise ValueError(f"Unknown transform class: {class_name}")

        else:
            # No parameters, just instantiate by class name
            if transform_item in globals():
                parsed_transforms.append(globals()[transform_item]())
            else:
                raise ValueError(f"Unknown transform class: {transform_item}")
    
    return parsed_transforms if len(parsed_transforms) > 1 else parsed_transforms[0]



def clean_lag_transforms(lag_transforms):
    """Converts lag transforms dictionary into a readable string identifier."""
    if not lag_transforms:
        return "No_Lag_Transforms"
    
    transform_names = []
    for lag, funcs in lag_transforms.items():
        func_names = "_".join(func.__name__ for func in funcs)
        transform_names.append(f"Lag{lag}:{func_names}")
    
    return "|".join(transform_names)  # Join using "|" for readability


# def save_results(results, filename="forecast_results.json"):
#     """Serializes model results into JSON format for easy reloading."""
#     serializable_results = {
#         json.dumps({
#             "Model": model,
#             "Transforms": transforms,
#             "Lags": list(lags),
#             "Lag Transforms": lag_transforms,
#             "Lag Name": lag_name
#         }): mape
#         for (model, transforms, lags, lag_transforms, lag_name), mape in results.items()
#     }
    
#     with open(filename, "w") as f:
#         json.dump(serializable_results, f, indent=4)
#     print(f"Results saved to {filename}")

def save_results(results, filename="forecast_results.json"):
    """Serializes model results into JSON format for easy reloading."""
    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def load_results(filename="forecast_results.json"):
    """Loads results from JSON and reconstructs into a structured DataFrame."""
    with open(filename, "r") as f:
        loaded_results = json.load(f)
    
    unpacked_results = []
    for key, mape_metr in loaded_results.items():
        result_data = json.loads(key)  # Convert back from JSON string
        
        unpacked_results.append([
            result_data["Model"],
            result_data["Transforms"],
            tuple(result_data["Lags"]),  # Convert back to tuple
            result_data["Lag Transforms"],
            result_data["Lag Name"],
            mape_metr
        ])
    
    # Convert into DataFrame
    df_results = pd.DataFrame(unpacked_results, columns=["Model", "Transforms", "Lags", "Lag Transforms", "Lag Name", "MAPE"])
    return df_results
