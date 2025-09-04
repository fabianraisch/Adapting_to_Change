
from src.data.data_preprocessing import preprocess_data_for_evaluation
from src.utils.eval_util import eval_on_dataloader, get_metric_function_from_string

from src.utils.model_util import load_scaler
from src.models.NaiveModel import NaiveModel

def evaluate_model(
        pretrained_model_dict = None,
        config: dict = {},
        eval_metrics: list[str] = [],
        device: str = "cuda",
		is_test: bool = False,
		return_tensors: bool = False
):
	"""
    Evaluate a pretrained model on the given dataset using the specified evaluation metrics.

    Args:
        pretrained_model (str): Name of the pretrained model.
        config (dict): Configuration dictionary for data loading, model, etc.
        eval_metrics (list[str]): List of metric names as strings to evaluate the model.
        device (str): The device to run the model on. Default is "cuda".
        is_test (bool): If True, evaluation is done on the test split. Otherwise, full data is used.
        return_tensors (bool): If True, also return a dictionary with actual and predicted tensors.

    Returns:
        dict: Dictionary with evaluation metric results.
        If return_tensors is True, also returns:
        dict: Dictionary with keys "pred" and "actuals" containing the respective tensors.
    """
	
	model = pretrained_model_dict["model"]
	loaded_cfg = pretrained_model_dict["config"]
	scaler_features, scaler_target = load_scaler()

	# Load Model and Scalers

	if config.get("model"): 
		config["model"].update(loaded_cfg["model"])
	else: 
		config["model"] = loaded_cfg["model"]


	# Push model onto the desired device
	model.to(device=device)
	if is_test:
		data_loader = preprocess_data_for_evaluation(
        forecast_horizon=config["model"]["forecast_horizon"],
        scaler_features=scaler_features,
        scaler_target=scaler_target,
		is_test=True,
        **config["data"],
        **config["dataloader"]
        )
	else:
		data_loader = preprocess_data_for_evaluation(
			forecast_horizon=config["model"]["forecast_horizon"],
			scaler_features=scaler_features,
			scaler_target=scaler_target,
			**config["data"],
			**config["dataloader"]
			)
	
	# Evaluate the model on the dataloader
	actuals_tensor, predictions_tensor = eval_on_dataloader(
		model=model,
		device=device,
		dataloader=data_loader,
		scale_target=loaded_cfg["data"]["scale_target"],
		scaler_target=scaler_target,

	)
	naive_act_tensor, naive_pred_tensor = eval_on_dataloader(
		model=NaiveModel(config["model"]["forecast_horizon"]),
		device=device,
		dataloader=data_loader,
		scale_target=loaded_cfg["data"]["scale_target"],
		scaler_target=scaler_target
	)
	metric_results = {}

	# Iterate over all metric and evaluate the model
	for metric in eval_metrics:
		
		metric_fn = get_metric_function_from_string(metric)

		if not metric_fn:
			break

		metric_result = metric_fn(actuals_tensor, predictions_tensor, naive_act_tensor, naive_pred_tensor)

		metric_results[metric] = metric_result

	if return_tensors:
		return metric_results, {"actuals": actuals_tensor, "pred": predictions_tensor}
	else:
		return metric_results