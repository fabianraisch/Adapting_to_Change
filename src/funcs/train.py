from src.utils.utilfunctions import read_config_from_json_string
from src.utils.model_util import get_model_from_string, delete_best_model_artifact
from src.data.data_preprocessing import preprocess_data
from src.models.NaiveModel import NaiveModel
from src.utils.train_util import train_epochs
from src.utils.adaptive_learning.gem import GEM
from src.utils.utilfunctions import override_model

from src.utils.model_util import load_scaler

import os
import torch
import random
from copy import deepcopy

def train(
        pretrained_model_dict = None,    # Give pretrained model here in format {model, optimizer, grad_scaler, config} as dict
        config: dict = {},
        tmp_model_save_folder: str = os.path.join("tmp", "best_models"),
        device: str = "cuda",
        ewc_dict: list = None,       # EWC dict. If empty, no EWC is used (default). Else should be contain keys "importance" and "ewc_list".
        rpr_dict: dict = None,
        gem: GEM = None
):

    train_id = str(random.randint(1e8, 1e9-1))
    
    # From now on work with a copied config variable to avoid sudden bugs.
    config_ = deepcopy(config)

    model_exists = pretrained_model_dict

    if not model_exists: # Create new model
        print("Found no model to load -> Creating new model from scratch")
        # Check if there is a section in the config dict that is called "model". If not, a new model cannot be created.
        if not config_.get("model"): 
            raise ValueError("Error: A model dict must be given when training a new ML model.")
        
        model_class = get_model_from_string(config_["model"]["name"])

        # The case that the model name string is not mapped to a ML model in the utils functions. 
        if model_class is None:
            raise ValueError(f"No model with string {config_['model']['name']} found. Please provide a name that is mapped to a model.")

        # Initialize the model from the model class.
        model = model_class(
            num_features=len(config_["data"]["feature_cols"]),
            seq_len=config["data"]["seq_len"],      # Needed for sequeeze operations for example in MLP.
            num_targets=len(config_["data"]["target_cols"]),
            **config_["model"]
            )

        if not config["model"].get("weight_decay"):
            config["model"]["weight_decay"] = 0.0
        
        # Initialize optimizer
        optimizer = config_["model"]["optimizer"](
            model.parameters(), 
            lr=config["model"]["lr"],
            weight_decay=config["model"]["weight_decay"]
            )
        
        # Initialize GradScaler
        grad_scaler = torch.amp.GradScaler("cuda", enabled=config_["training"]["use_amp"])

        scaler_features = None
        scaler_target = None

    else:   # Load a model
        loaded_dict = deepcopy(pretrained_model_dict)

        model = loaded_dict["model"]
        optimizer = loaded_dict["optimizer"]
        grad_scaler = loaded_dict["grad_scaler"]

        # Model config update. The model config of the general model is used instead of the given model config in the JSON when finetuning.
        loaded_cfg = read_config_from_json_string(loaded_dict["config"])
        if not config_.get("model"): 
            config_["model"] = {}
        config_["model"].update(loaded_cfg["model"])
        config_["data"]["feature_cols"] = loaded_cfg["data"]["feature_cols"]
        config_["data"]["target_cols"] = loaded_cfg["data"]["target_cols"]
        config_["data"]["seq_len"] = loaded_cfg["data"]["seq_len"]

        scaler_features, scaler_target = load_scaler()

    model.to(device=device)

    if gem:         # Set the model for GEM if used.
        gem.set_model(model=model)

    # Check if there should be overrides in the model.
    if config_.get("override") and config_["override"]["do_override"]:
        model, optimizer = override_model(model=model,
                       optimizer=optimizer, 
                       **config_["override"])

    train_data_loader, val_data_loader, test_data_loader, scaler_features, scaler_target = preprocess_data(
        forecast_horizon=config_["model"]["forecast_horizon"],
        scaler_features=scaler_features,
        scaler_target=scaler_target,
        **config_["data"],
        **config_["dataloader"]
        )
    
    dataloaders = (train_data_loader, val_data_loader)

    # Create naive model for MASE calculation
    naive_model = NaiveModel(forecast_horizon=config_["model"]["forecast_horizon"], num_target_cols=len(config_["data"]["target_cols"]))

    # Training loop over the epochs
    val_eval_dict, best_model = train_epochs(model=model,
                 optimizer=optimizer,
                 grad_scaler=grad_scaler,
                 dataloaders=dataloaders,
                 tmp_model_save_folder = tmp_model_save_folder,
                 scale_target = config_["data"]["scale_target"],
                 scaler_target = scaler_target,
                 train_id=train_id,
                 device=device,
                 config=config_,
                 naive_model=naive_model,
                 do_initial_evaluation=model_exists,
                 ewc_dict=ewc_dict,
                 rpr_dict=rpr_dict,
                 gem = gem,
                 **config_["training"])
    
    
    best_model_path = os.path.join(tmp_model_save_folder, train_id + ".pt")

    
    # Model save
    if best_model:
        model = best_model["model"]
        optimizer = best_model["optimizer"]
        grad_scaler = best_model["grad_scaler"]

    # Cleanup best model
    delete_best_model_artifact(
        best_model_path=best_model_path
    )


    return val_eval_dict, {"model": model, "optimizer": optimizer, "grad_scaler": grad_scaler, "config": config}
