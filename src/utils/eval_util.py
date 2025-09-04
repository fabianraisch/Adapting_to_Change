from src.eval_metrics.eval_metrics import rmse_metric, mae_metric, mase_metric

import torch

def get_metric_function_from_string(s: str):
    
    s = s.lower()

    if s == "rmse": 
        return rmse_metric
    if s == "mae": 
        return mae_metric
    if s == "mase":
        return mase_metric


def eval_on_dataloader(
        model = None,
        device: str = "cpu",
        dataloader = None,
        scale_target: bool = False,
        scaler_target = None
):
    
    model.eval()
    with torch.no_grad():

        # Takes all the actuals and predictions from the dataloaders so they can be evaluated at once.
        actuals_ = []
        predictions_ = []
        for X_batch, y_batch in dataloader:
            actuals_.append(y_batch.to(device))
            predictions_.append(model(X_batch.to(device)))
            
        actuals = torch.cat(actuals_)
        predictions = torch.cat(predictions_)

        # Inverse scaling
        if scale_target: 
            actuals_inverse_scaled = scaler_target.inverse_transform(actuals.cpu().numpy().reshape(-1, model.num_targets)).flatten()
            predictions_inverse_scaled = scaler_target.inverse_transform(predictions.cpu().numpy().reshape(-1, model.num_targets)).flatten()
        else: 
            actuals_inverse_scaled = actuals.cpu().numpy().reshape(-1, 1).flatten()
            predictions_inverse_scaled = predictions.cpu().numpy().reshape(-1, 1).flatten()

        # Convert back to tensors for loss calculation
        actuals_tensor = torch.tensor(actuals_inverse_scaled, dtype=torch.float32)
        predictions_tensor = torch.tensor(predictions_inverse_scaled, dtype=torch.float32)

        return actuals_tensor, predictions_tensor
    


def evaluate_epoch(
        model = None,
        device: str = "cpu",
        dataloader = None,
        scale_target: bool = False,
        scaler_target = None,
        naive_model = None,
        loss_function = None,
        evaluation_type: str = "train",
        epoch_number: int = -1
):

    # Do the Evaluation process for the epoch
    actuals_tensor, predictions_tensor = eval_on_dataloader(
        model=model,
        device=device,
        dataloader=dataloader,
        scale_target=scale_target,
        scaler_target=scaler_target
    )

    # In case there exists a Naive model, calculate the MAE of it in this Epoch
    if naive_model:
        naive_act_tensor, naive_pred_tensor = eval_on_dataloader(
            model=naive_model,
            device=device,
            dataloader=dataloader,
            scale_target=scale_target,
            scaler_target=scaler_target,
        )

    # Calculate RMSE
    rmse = torch.sqrt(loss_function(predictions_tensor, actuals_tensor)).item()

    # Calculate MAEs
    mae_fun = torch.nn.L1Loss()

    mae = mae_fun(predictions_tensor, actuals_tensor).item()

    # Calculate the MASE if there exists a Naive Model
    if naive_model:
        naive_mae = mae_fun(naive_pred_tensor, naive_act_tensor).item()
        mase = mae / naive_mae

    return_dict = {"rmse": rmse, "mae": mae}

    if naive_model: 
        return_dict["mase"] = mase

    return return_dict
