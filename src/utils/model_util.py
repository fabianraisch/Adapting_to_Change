import os
import joblib
import torch
from torch import nn, optim

from src.models.lstm import LSTM

from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load_scaler(
          feature_scaler_name = "feature_scaler",
          target_scaler_name = "target_scaler",
          dir = "models/scaler"
          ):

    fs_scaler_load_name = f"{dir}/{feature_scaler_name}.gz"
    ts_scaler_load_name = f"{dir}/{target_scaler_name}.gz"

    fs = joblib.load(fs_scaler_load_name)
    ts = joblib.load(ts_scaler_load_name)

    return fs, ts


def get_new_scaler_from_string(s: str = ""):

    s = s.lower()

    scaler_names = ["minmax", "standard"]

    if s == "minmax": 
        return MinMaxScaler(feature_range=(0, 1))
    if s == "standard": 
        return StandardScaler()

    raise ValueError(f"Scaler name {s} unknown -> please specify an available scaler name from the list {scaler_names}")


def get_model_from_string(s: str = ""):
    '''
    Function to map an identifying string to a model class (here nn.Module).

    Args: 
        - s [str]: model identifying string

    Returns:
        - model class for the corresponding string. If no model is found, this is None.
    '''

    s = s.lower()

    if s == "lstm": 
        return LSTM

    return None


def load_model(path = "models/model.pt", device="cuda"):
        
    saved_dict = torch.load(path, weights_only=False)
    
    saved_cfg = saved_dict.get("config", {})

    if not saved_cfg:
         raise ValueError("No config was found in load model -> Config needed to reproduce the ML model.")
    
    model_string = saved_cfg.get("model", {}).get("name")

    if not model_string:
         raise ValueError("No model string could be found in the loaded config.")
    
    model_class = get_model_from_string(model_string)

    # Load the model
    model = model_class(
            num_features=len(saved_cfg["data"]["feature_cols"]),
            seq_len=saved_cfg["data"]["seq_len"],
            num_targets=len(saved_cfg["data"]["target_cols"]),
            **saved_cfg["model"]
            )
    
    model.load_state_dict(saved_dict["model"])

    model.to(device=device)
    
    # Load the Optimizer [Note: Weight decay is L2-Regularization]
    if saved_cfg.get(model) and saved_cfg["model"].get("weight_decay"):
        optimizer = saved_cfg["model"]["optimizer"](
            model.parameters(), 
            lr=saved_cfg["model"]["lr"],
            weight_decay= saved_cfg["model"]["weight_decay"]
            )
    else:
        optimizer = saved_cfg["model"]["optimizer"](
            model.parameters(), 
            lr=saved_cfg["model"]["lr"]
            )
    
    optimizer.load_state_dict(saved_dict["optimizer"])

    # Load the Grad Scaler
    grad_scaler = torch.amp.GradScaler("cuda", enabled=saved_cfg["training"]["use_amp"])

    grad_scaler.load_state_dict(saved_dict["grad_scaler"])
    
    return {
         "model": model,
         "optimizer": optimizer,
         "grad_scaler": grad_scaler,
         "config": saved_cfg
    }


def delete_best_model_artifact(
          best_model_path: str = ""
):
     
     if os.path.exists(best_model_path):
         
         os.remove(best_model_path)


def best_model_exists(
     best_model_tmp_path: str = "",
     train_id: str = "1",
):
     
    checkpoint_name = train_id + ".pt"

    return os.path.exists(os.path.join(best_model_tmp_path, checkpoint_name))


def save_current_best_model(
        model: nn.Module = None,
        optimizer: optim.Optimizer = None,
        grad_scaler = None,
        config = {},
        device = "cpu"
):
        
        # Convert model to cpu so its properly serializable
        model.to(torch.device("cpu"))
        
        checkpoint = {
            'model': model,
            'optimizer': optimizer,
            'grad_scaler': grad_scaler,
            'config': config
        }
        
        # Convert the model back to the device from before
        model.to(device)

        return checkpoint