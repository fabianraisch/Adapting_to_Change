
import torch
from torch import nn
from sklearn.metrics import mean_absolute_percentage_error


def rmse_metric(predicted, actual, *args, **kwargs):

    mse_loss_fn = nn.MSELoss()

    mse_loss = mse_loss_fn(predicted, actual)

    return torch.sqrt(mse_loss).item()


def mae_metric(predicted, actual, *args, **kwargs):
    mae_fun = torch.nn.L1Loss()

    return mae_fun(predicted, actual).item()

def mase_metric(predicted, actual, naive_act_tensor,  naive_pred_tensor, **kwargs):
    mae = mae_metric(predicted, actual)
    naive_mae = mae_metric(naive_pred_tensor, naive_act_tensor)
    mase_value = mae / naive_mae
    
    return mase_value

def mape_metric(predicted, actual, *args, **kwargs):
    # calculate mean absolute percentage error.
    # Scikits docs mention a parameter 'epsilon', a very small number that replaces the denominator should it get too small
    return mean_absolute_percentage_error(actual, predicted)



