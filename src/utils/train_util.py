from src.utils.model_util import save_current_best_model
from src.utils.eval_util import evaluate_epoch
from src.utils.adaptive_learning.ewc import EWC
from src.utils.adaptive_learning.gem import GEM
from src.utils.adaptive_learning.rpr import reinitialize_parameters

import torch
from torch import nn, optim


#--------------------------------------------------
#           Main Train Epochs Function
#--------------------------------------------------
def train_epochs(
        model: nn.Module = None,                # Model to be trained
        optimizer: optim.Optimizer = None,      # Optimizer that is used to train
        grad_scaler = None,                     # Grad scaler used for autocast. If autocast is disabled, this is also disabled and a no-op.
        dataloaders: list = [],                 # List of dataloaders, contains train and test loader.
        use_amp: bool = True,                   # Indicates if amp is used (automatic mixed precision).
        tmp_model_save_folder = "",             # Folder to save the temporary best model.
        epochs= 5,                              # Number of epochs to train
        device=None,                            # Device to train on. Either "cpu" or "cuda", or "cuda:0" with the int indicating on which GPU to train on.
        loss_function= nn.MSELoss(),            # The loss function this model is trained on.
        scale_target: bool = True,              # Indicates whether targets are scaled
        scaler_target = None,                   # A target scaler if exists
        naive_model: nn.Module = None,          # Naive model to calculate the MASE.
        save_best_model="",                     # {train/val}. Save best model of those category. If None or empty, save last model.
        save_best_metric = "",                  # {rmse/mae}. Save best model according to the selected metric. If None or empty, use rmse.
        config = {},                            # Config part for training.
        train_id = "1",                         # Id of the training process.
        do_initial_evaluation = False,          # Boolean to make an initial evaluation. If set to true, the model will be evaluated before (re-)training. 
        ewc_dict = {},                          # Dict of previous datasets to use for EWC and the importance. If EWC isn't in use, this can be left empty.
        rpr_dict = {},                          # Dict of options for Random Parameter Reinitialization. If RPR isn't in use, this can be left empty.
        gem: GEM = None,                        # Gradient Episodic Memory object that contains the previous gradients.
        **kwargs                                # Other arguments for training.
):
    
    best_model = None
    
    batch_count=0
    best_model_errors = {
        "train_rmse": float("inf"),
        "val_rmse": float("inf"),
        "train_mae": float("inf"),
        "val_mae": float("inf"),
        "train_mase": float("inf"),
        "val_mase": float("inf"),
    }
    train_loader, val_loader = dataloaders
    save_best_train = save_best_model == "train"
    save_best_val = save_best_model == "val"
    if not save_best_metric:
        save_best_metric = "rmse"

    # Check if ewc_dict is configured right
    if ewc_dict:
        ewc = EWC(model=model, 
                  dataset=ewc_dict["datasets"], 
                  is_disabled=len(ewc_dict["datasets"]) == 0, 
                  use_cuda = device == "cuda", 
                  loss_function=loss_function)

    # If RPR specified, reinitialize proportion of model weights randomly.
    if rpr_dict:
        reinitialize_parameters(
                    model,
                    reinit_prop=rpr_dict["reinit_prop"],
                    device=device
                )

    # Train the model epoch by epoch
    for epoch in range(-1,epochs):

        ## Training process for the epoch
        if epoch >= 0:

            model.train() # enter training mode

            for x_batch, y_batch in train_loader:
                batch_count += 1
                
                #Check dim of y_batch and make it 3d for multivariate time series
                if y_batch.dim() == 2:
                    y_batch = y_batch.unsqueeze(-1)
                
                # Convert to device
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                # Normal - Autocast used if specified in the constructur
                with torch.autocast(device_type="cuda",dtype=torch.float16, enabled=use_amp):

                    loss = torch.tensor(0.0, requires_grad=False)   # Initialize Torch Loss.

                    # In case EWC is available, calculate its loss and add it to the standard loss.
                    if ewc_dict:
                        loss = loss + ewc_dict["importance"] * ewc.penalty(model=model)
                        optimizer.zero_grad()

                    # Normal prediction and loss calculation
                    predicted = model(x_batch)
                    loss =  loss + loss_function(predicted, y_batch)


                # Optimizer and Backprop workflow
                optimizer.zero_grad()

                # If gem exists, project the gradients.
                if gem:
                    gem.observe(x=x_batch, y=y_batch)

                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

        ## Evaluation process for the epoch
        if (epoch >= 0) or do_initial_evaluation:
            # If do_initial_evaluation is true, we first want to evaluate the model before (re-)training.
            if epoch == -1:
                print("Doing an initial evaluation step before training.")

            # Evaluate over the whole train set
            train_eval_dict = evaluate_epoch(
                model=model,
                device=device,
                dataloader=train_loader,
                scale_target=scale_target,
                scaler_target=scaler_target,
                naive_model=naive_model,
                loss_function=loss_function,
                evaluation_type="train",
                epoch_number=epoch
            )

            # Evaluate over the whole validation set
            val_eval_dict = evaluate_epoch(
                model=model,
                device=device,
                dataloader=val_loader,
                scale_target=scale_target,
                scaler_target=scaler_target,
                naive_model=naive_model,
                loss_function=loss_function,
                evaluation_type="eval",
                epoch_number=epoch
            )

            # Check if model should be saved (based on train error)
            if save_best_train and (best_model_errors[f"train_{save_best_metric}"] > train_eval_dict[save_best_metric]):
                print(f"--> {save_best_metric.upper()} decreased - model will be saved as best model.")
                best_model_errors["train_rmse"] = train_eval_dict["rmse"]
                best_model_errors["train_mae"] = train_eval_dict["mae"]
                best_model_errors["train_mase"] = train_eval_dict["mase"]
                best_model_errors["val_rmse"] = val_eval_dict["rmse"]
                best_model_errors["val_mae"] = val_eval_dict["mae"]
                best_model_errors["val_mase"] = val_eval_dict["mase"]
                best_model = save_current_best_model(
                    model=model,
                    optimizer=optimizer,
                    grad_scaler=grad_scaler,
                    config=config,
                    device=device
                )

            # Check if model should be saved (based on val error)
            if save_best_val and len(val_loader) and (best_model_errors[f"val_{save_best_metric}"] > val_eval_dict[save_best_metric]):
                print(f"--> {save_best_metric.upper()} decreased - model will be saved as best model.")
                best_model_errors["train_rmse"] = train_eval_dict["rmse"]
                best_model_errors["train_mae"] = train_eval_dict["mae"]
                best_model_errors["train_mase"] = train_eval_dict["mase"]
                best_model_errors["val_rmse"] = val_eval_dict["rmse"]
                best_model_errors["val_mae"] = val_eval_dict["mae"]
                best_model_errors["val_mase"] = val_eval_dict["mase"]
                best_model = save_current_best_model(
                    model=model,
                    optimizer=optimizer,
                    grad_scaler=grad_scaler,
                    config=config,
                    device=device
                )

    if save_best_train or save_best_val: 
        return best_model_errors, best_model

    return {"train_rmse": train_eval_dict["rmse"], 
            "val_rmse": val_eval_dict["rmse"], 
            "train_mae": train_eval_dict["mae"], 
            "val_mae": val_eval_dict["mae"],
            "train_mase": train_eval_dict.get("mase"),
            "val_mase": val_eval_dict.get("mase")}, model
