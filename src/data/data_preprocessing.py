from src.utils.utilfunctions import read_all_dataframes_from_dir
from src.utils.model_util import get_new_scaler_from_string

import torch
from typing import Union
from typing import List
import pandas as pd
from torch.utils.data import DataLoader
import random
from copy import deepcopy
import re

def preprocess_data_for_evaluation(
        folder:str = "",
        seq_len: int = 96,
        train_split: float = 1.0,
        val_split: float = 0.0,
        dataframe_limit: int = None,
        dataframe_start: int = None,
        target_cols: Union[list, str] = "",
        feature_cols: Union[list, str] = [],
        scale_target: bool = True,
        scaler_features = None,
        scaler_target = None,
        feature_scaler_type = "",
        target_scaler_type = "",
        forecast_horizon: int = 1,
        batch_size: int = 16,
        handle_nan: str = "",
        is_test: bool = False,
        **kwargs
):
    '''
    Function for data preprocessing only for the evaluation step.
    Variables are documented in the "preprocess_data" function.
    '''
    if not is_test:
        train_split = 1.0
        val_split = 0.0
    train_dataloader, _, test_dataloader, _, _= preprocess_data(
        folder=folder,
        seq_len=seq_len,
        train_split=train_split,
        val_split=val_split,
        train_ranges=[],
        train_ranges_col_name="",
        dataframe_limit=dataframe_limit,
        dataframe_start=dataframe_start,
        target_cols=target_cols,
        feature_cols=feature_cols,
        scale_target=scale_target,
        scaler_features = scaler_features,
        scaler_target = scaler_target,
        feature_scaler_type = feature_scaler_type,
        target_scaler_type = target_scaler_type,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size,
        handle_nan=handle_nan,
        shuffle=False,
    )
    if is_test:
        return test_dataloader
    else:
        return train_dataloader



def preprocess_data(
        folder: str = "",
        seq_len: int = 96,
        train_split: float = 0.0,               # Ratio Split on Train data, the rest is used for validation data.
        val_split: float = 0.0,                 # Ratio Split on Validation data, the rest is used for test data.
        train_ranges: list = [],                # Ranges of train sets given a specific column.
        val_ranges: list = [],                  # Ranges of validation sets given a specific column.
        train_ranges_col_name: str = "",        # Column name of the train ranges.
        series_ratio: float = 0.0,              # Alternative data split timeseries-wise. This fraction of timeseries is used for train, the rest val.
        val_series: List[str] | str = [],       # String or List of strings, which will be regex-searched on the timeseries given. Those matches will be put into validation set.
        dataframe_limit: int = None,            # Cuts off the dataframe after dataframe_limit rows. 
        dataframe_start: int = None,            # Start each dataframe at row dataframe_start.
        target_cols: Union[list, str] = "",     # Columns that are target variables.
        feature_cols: Union[list, str] = [],    # Columns of the feature variables. Does not have to contain the target variables with lag of 1.
        scale_target: bool = True,              # Indicates on whether the target variables will be scaled
        scaler_features = None,                 # Variable for a pre-fitted feature scaler
        scaler_target = None,                   # Variable for a pre-fitted target scaler
        feature_scaler_type = "",               # Type of a newly created feature scaler
        target_scaler_type = "",                # Type of a newly created target scaler
        forecast_horizon: int = 1,              # Forecast horizon of the predictions
        batch_size: int = 16,                   # Number of training examples in a batch of the dataloader
        exclude: list = [],                     # Choose a list of keywords of CSVs that will be excluded if they contain those strings.
        shuffle: bool = True,                   # Shuffle the data in the dataloader
        handle_nan: str = "",                   # String indicating how to handle NaN values in the timeseries given.
        **kwargs):

        # Read the raw dataframes from the file directory.
        raw_dfs = read_all_dataframes_from_dir(
            csv_file_dir = folder,
            exclude = exclude,
            dataframe_limit=dataframe_limit,
            dataframe_start=dataframe_start,
            )
        
        # Check if any CSV files were found.
        if not raw_dfs: 
            raise ValueError(f"No CSV files were found in the folder {folder}. Maybe you forgot the '_' prefix?")

        # Make Splits
        train_list, vals_list, test_list = [], [], []

        if train_ranges or train_split: 
            for df in raw_dfs.values():
                if train_ranges:
                    if train_ranges_col_name:
                        trains, vals = _make_ranges_split(df, train_ranges, val_ranges, train_ranges_col_name)
                        test = []  # Ensure test is an empty list if not provided
                    else:
                        trains, vals = _make_ranges_split(df, train_ranges, val_ranges, None)
                        test = []  # Ensure test is an empty list if not provided
                elif train_split:
                    trains, vals, test = _make_ratio_split(df, train_split, val_split) 
                else: 
                    raise ValueError("Neither train_timestamp ranges nor train_split is given. Please specify one in the config.")
                train_list.extend(trains)
                vals_list.extend(vals)
                test_list.extend(test)

        # Note. Serires ratio does not work and is not intended to work with train-val-test split, only with train-val split.
        elif series_ratio:
            df_values = list(raw_dfs.values())
            random.shuffle(df_values)

            trains, vals = _make_series_ratio_split(df_values, series_ratio)

            train_list.extend(trains)
            vals_list.extend(vals)

        # Case val serires exist, make regex pattern match on the csv names.
        elif val_series:

            if isinstance(val_series, str):
                val_series = [val_series]

            train_strings, val_strings = [], []

            for k, v in raw_dfs.items():

                # Check by regex whether to put the dataframe into train or val.
                if any(re.search(pattern, k) for pattern in val_series):
                    vals_list.append(v)
                    val_strings.append(k)
                else:
                    train_list.append(v)
                    train_strings.append(k)

        else: 
            raise ValueError("Neither train_timestamp ranges nor train_split is given not split_ratio. Please specify one in the config.")

        # Preprocess train, validation, and test dfs. We treat each split just like a new timeseries.
        preprocessed_trains, preprocessed_vals, preprocessed_tests = [], [], []

        for idx, df in enumerate(train_list):
            preprocessed_df = _preprocess_df(
                idx=idx,
                df=df
            )

            preprocessed_trains.append(preprocessed_df)

        for idx, df in enumerate(vals_list):
            preprocessed_df = _preprocess_df(
                idx=idx,
                df=df
            )

            preprocessed_vals.append(preprocessed_df)

        for idx, df in enumerate(test_list):
            preprocessed_df = _preprocess_df(
                idx=idx,
                df=df
            )

            preprocessed_tests.append(preprocessed_df)

        # Create Target Scaler if not already exists
        if (not scaler_target) and scale_target:
            scaler_target = get_new_scaler_from_string(target_scaler_type)
            _fit_scaler_on_cols(scaler_target, preprocessed_trains, target_cols)

        # Create Feature Scaler if not already exist
        if not scaler_features:
            scaler_features = get_new_scaler_from_string(feature_scaler_type)
            _fit_scaler_on_cols(scaler_features, preprocessed_trains, feature_cols)

        # Scale Features
        _apply_scaler_on_dataframes(scaler_features, preprocessed_trains, feature_cols)
        _apply_scaler_on_dataframes(scaler_features, preprocessed_vals, feature_cols)
        _apply_scaler_on_dataframes(scaler_features, preprocessed_tests, feature_cols)
        # Scale all targets that are not already contained in the features.
        if scale_target:
            # Note: Right now this only works for 1 target column. If more are added, this needs to be modified slightly.
            unique_targets = [t for t in target_cols if t not in feature_cols]
            if unique_targets:
                _apply_scaler_on_dataframes(scaler_target, preprocessed_trains, unique_targets)
                _apply_scaler_on_dataframes(scaler_target, preprocessed_vals, unique_targets)
                _apply_scaler_on_dataframes(scaler_target, preprocessed_tests, unique_targets)  
            

        # Preprocess the datasets with seq_len and forecast_horizon into "snippets"
        train_sets, val_sets, test_sets = [], [], []

        for idx, train_df in enumerate(preprocessed_trains):
            train_set = _transform_dataset_multivariate(
                        data_X=torch.tensor(train_df[feature_cols].astype('float32').values),
                        data_Y=torch.tensor(train_df[target_cols].astype('float32').values),
                        seq_len=seq_len,
                        forecast_horizon=forecast_horizon,
                        nan_handle=handle_nan
                    )
            train_sets.extend(train_set)

        for idx, val_df in enumerate(preprocessed_vals):
            val_sets = _transform_dataset_multivariate(
                        data_X=torch.tensor(val_df[feature_cols].astype('float32').values),
                        data_Y=torch.tensor(val_df[target_cols].astype('float32').values),
                        seq_len=seq_len,
                        forecast_horizon=forecast_horizon,
                        nan_handle=handle_nan
                    )    
            val_sets.extend(val_sets)   

        for idx, test_df in enumerate(preprocessed_tests):
            test_sets = _transform_dataset_multivariate(
                        data_X=torch.tensor(test_df[feature_cols].astype('float32').values),
                        data_Y=torch.tensor(test_df[target_cols].astype('float32').values),
                        seq_len=seq_len,
                        forecast_horizon=forecast_horizon,
                        nan_handle=handle_nan
                    )
            test_sets.extend(test_sets)
        train_dataloader = DataLoader(train_sets, batch_size=batch_size, shuffle=shuffle)
        
        #Val-Dataloader may also be empty
        try:
            val_dataloader = DataLoader(val_sets, batch_size=batch_size, shuffle=False)
        except ValueError:
            val_dataloader= None
        try:
            test_dataloader = DataLoader(test_sets, batch_size=batch_size, shuffle=False)
        except ValueError:
            test_dataloader= None

        return train_dataloader, val_dataloader, test_dataloader, scaler_features, scaler_target


def _fit_scaler_on_cols(scaler, dfs, cols):

    df_concat = pd.concat(dfs, ignore_index=True)

    scaler.fit(df_concat[cols])


def _apply_scaler_on_dataframes(scaler, dfs, cols):

    for df in dfs:
        
        if df.empty: 
            continue

        new_cols = scaler.transform(df[cols])
        df[cols] = new_cols


def _make_series_ratio_split(dfs: List[pd.DataFrame], series_ratio:int = 0.7):
    """
    Function to make serires ratio split on a list of dataframes given a ratio. A ratio (given) of all timeseries given in the list are returned for train, the rest is val.

    Args:
    - dfs: List of dataframes to be split
    - series_ratio: Ratio of the dataframes that should be assigned to the train set. For instance, if ratio is 0.7, 70% are train, 30% are in val.

    Returns: Train dataframes and val dataframes.
    """

    random.shuffle(dfs)

    num_train_series = int(len(dfs) * series_ratio)

    return dfs[:num_train_series], dfs[num_train_series:]



def _make_ratio_split(df, train_ratio, val_ratio=0.0):
    """
    Function to split a dataframe into train, validation, and test sets based on ratios.

    Args:
    - df: the dataframe to be split
    - train_ratio: the ratio of the dataset to be used for training.
    - val_ratio: the ratio of the dataset to be used for validation.

    Returns: a list of train, validation, and test sets.
    """
    if val_ratio == 0.0:
        # If no validation ratio is given, use the rest of the data for validation
        train_end = int(len(df) * train_ratio)
        train = [df[:train_end].copy()]
        val = [df[train_end:].copy()]
        test = []
    else:
        train_end = int(len(df) * train_ratio)
        val_end = train_end + int(len(df) * val_ratio)
        train = [df[:train_end].copy()]
        val = [df[train_end:val_end].copy()]
        test = [df[val_end:].copy()]

    return train, val, test


def _handle_nan(x, y, nan_handle: str):
    '''
    Function to handle NaN entries in timeseries snippets of the dataloader given.

    Args:
    - x: x values of the timeseries.
    - y: y values of the timeseries (targets).
    - nan_handle: string indicating how to handle NaN values / rows in the dataloader.

    Returns: The dataloader. Changes are made in-place. 
    '''

    # In case no option is given.
    if not nan_handle:
        return x, y
    
    # Skips all datasets that contain a NaN value in it.
    if nan_handle == "skip":

        if torch.isnan(x).any() or torch.isnan(y).any():

            return None, None
        
        return x, y


def _make_ranges_split(df, train_ranges, val_ranges, ranges_col_name):
    '''
    Function to split a dataframe into a list of train sets and val sets.

    Args:
    - df: the dataframe to be split
    - train_ranges: the ranges of the train sections. All sections left will be in the validation set.
    - val_ranges: the ranges of the validation sections. If not given, it will be constructed from the train ranges.
    - ranges_col_name: the name of the column the ranges should be split by.

    Returns: a list of train and validation sets.
    '''

    train = []
    val = []

    # If no ranges_col_name is given, add one to a dataframe copy.
    if not ranges_col_name:
        df = deepcopy(df)
        df.insert(0, "ranges_index", df.index)
        ranges_col_name = "ranges_index"


    def construct_df(df, start, end):
        to_return = df[(df[ranges_col_name] >= start) & (df[ranges_col_name] <= end)].copy()
        # Drop dummy index if exists
        to_return.drop(columns=["ranges_index"], errors="ignore", inplace=True)
        
        return to_return
    

    if not val_ranges:
        val_ranges = []

    train_ranges_sorted = sorted(train_ranges, key=lambda x: x[0])
    for i in range(1, len(train_ranges_sorted)):
        before_range = train_ranges_sorted[i-1]
        after_range = train_ranges_sorted[i]

        if before_range[1] > after_range[0]: 
            raise ValueError(f"Ranges are wrong in train timestep split: {before_range} and {after_range}")
        val_ranges.append([before_range[1] + 1, after_range[0] - 1])
    val_ranges.append([train_ranges_sorted[-1][1] + 1, float("inf")])
    for r in train_ranges:
        start = r[0]
        end = r[1]

        df_to_add = construct_df(df, start, end)
        if not df_to_add.empty: 
            train.append(construct_df(df, start, end))

    for r in val_ranges:
        start = r[0]
        end = r[1]

        df_to_add = construct_df(df, start, end)
        if not df_to_add.empty: 
            val.append(construct_df(df, start, end))
    _add_split_ids(train)
    _add_split_ids(val)

    
    return train, val


def _add_split_ids(dfs):

    for idx, df in enumerate(dfs):
        df["split_id"] = idx


def _preprocess_df(
        idx: int = 0,
        df: pd.DataFrame = None
):
    
    # Drop columns
    try: 
        df.drop(columns=['Unnamed: 0'], inplace=True)
    except Exception: 
        pass

    # Replace points as they are hard to process further down the streamline.
    df.columns = df.columns.str.replace('.', '')

    df.reset_index(inplace=True)

    return df


def _transform_dataset_multivariate(
        data_X, 
        data_Y,
        seq_len,
        forecast_horizon,
        nan_handle
        ):
    
    to_return = []

    for i in range(len(data_X) - seq_len - forecast_horizon + 1):
        window = i + seq_len
        x = data_X[i:window]
        y = data_Y[window:window + forecast_horizon].squeeze(-1)

        x, y = _handle_nan(x, y, nan_handle)

        # In case one of them is missing (i.e., some nan handling).
        if x is None or y is None:
            continue

        to_return.append([x, y])

    return to_return