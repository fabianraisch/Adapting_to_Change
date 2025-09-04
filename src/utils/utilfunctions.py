from torch import nn, optim
import json
import os
import pandas as pd
from typing import Dict


def override_model(
        model: nn.Module = None,
        optimizer = None,
        optimizer_override_type = None,
        optimizer_lr: float = None,
        **kwargs
        ):
    
    '''
    Function to handle all overrides in the finetuning phase.

    Args:
        - model: the model to be altered.
        - optimizer: the optimizer to be altered if needed.
        - optimizer_override_type: the type of the optimizer that should be overriden.
        - optimizer_lr: the learning rate of the newly created optimizer.

    Returns:
        - model: the altered model object.
        - optimizer: the potentially new optimizer with changed learning rate.
    '''
    
    # Optimizer LR Override if given
    if optimizer_override_type and optimizer_lr:

        print(f"Override the optimizer with a learning rate of {optimizer_lr}.")

        optimizer = optimizer_override_type(
            model.parameters(), 
            lr=optimizer_lr
            )
        

    return model, optimizer


def read_config(
        config_path: str = ""
):
    '''
    Function to read a config file and process some string. Processing strings means that identifying strings are transformed into the respective
    torch class. For example: optimizer = adam -> torch.nn.Adam()

    Args:
        - config_path [str]: the path to the config. Starts from ../configs/

    Returns:
        - Preprocessed config.
    '''

    # In case someone put an absolute path there.
    config_path = config_path.removeprefix("/")
    config_path = config_path.removeprefix("\\")

    with open(config_path, 'r') as f:
        config = json.load(f)

    config_formatted = read_config_from_json_string(config)
    
    return config_formatted


def read_config_from_json_string(
        config: str = ""
):
    
    if config.get("model"):
        model_cfg = config['model']

        if model_cfg.get("optimizer") == 'adam':
            model_cfg['optimizer'] = optim.Adam
        elif model_cfg.get("optimizer") == 'adamw':
            model_cfg['optimizer'] = optim.AdamW


    train_cfg = config.get('training')
    if train_cfg:
        if train_cfg['loss_function'] == 'mse':
                train_cfg['loss_function'] = nn.MSELoss()


    # Note: This might need some rework to make the optimizer type get more readable and usable.
    if config.get("override") and config["override"].get("do_override"):
        if config["override"]["optimizer_override_type"] == 'adam':
            config["override"]["optimizer_override_type"] = optim.Adam
    
    return config
    


def read_csv_data(
        filepath: str = "",
        sep: str = ",",
        dataframe_limit: int = None
):
    
    df = pd.read_csv(filepath, sep=sep)

    if dataframe_limit: 
        df = df[:dataframe_limit]

    return df


def read_all_dataframes_from_dir(
        csv_file_dir: str = "",
        exclude: list[str] = [], 
        dataframe_limit: int = None, 
        dataframe_start:int = None
        ) -> Dict[str, pd.DataFrame]:
        """
        Read all CSV files from a directory and its subdirectories that start with the given prefix into a list of DataFrames.

        Args:
            csv_file_dir (str): The directory containing the CSV files.
            dataframe_limit (int, optional): The maximum number of rows to read from each DataFrame. Defaults to None.
            dataframe_start (int, optional): The offset on where the dataframe should begin at.
            prefix (str, optional): Only CSV files starting with the prefix are read. Defaults to '_'.
            
        Returns:
            Dict[str, pd.DataFrame]: Dict of dataframes. Key: Name and Value: Dataframe.
        """

        def get_file_name(s):
            return os.path.basename(s).replace(".csv", "")
        
        dfs = {}

        # If a file is given in the path, only read this one file.
        if os.path.isfile(csv_file_dir):
            if csv_file_dir.endswith('.csv'):
                df = pd.read_csv(csv_file_dir)[dataframe_start : dataframe_limit]
                df_name = get_file_name(csv_file_dir)
                dfs[df_name] = df
            return dfs

        # This covers the case when a directory is given.
        for (root,_,files) in os.walk(csv_file_dir,topdown=True):
            for file in files:
                if file.endswith('.csv') and csv_file_dir.startswith('_'):
                    exclude_current = False
                    for pattern in exclude:
                        if pattern in file:
                            exclude_current = True

                    if exclude_current:
                        continue
                    path_to_csv = os.path.join(root, file)
                    df = pd.read_csv(path_to_csv)[dataframe_start : dataframe_limit]
                    df_name = get_file_name(path_to_csv)
                    dfs[df_name] = df

        return dfs