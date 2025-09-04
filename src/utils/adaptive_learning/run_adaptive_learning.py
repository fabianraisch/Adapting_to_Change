import os
from copy import deepcopy
import random
import pandas as pd

from src.funcs.train import train
from src.utils.adaptive_learning.gem import GEM
from src.utils.model_util import load_scaler, load_model
from src.funcs.evaluate import evaluate_model
from src.data.data_preprocessing import preprocess_data_for_evaluation


def run_adaptive_learning(config, device="cuda"):

    # set parameters
    year_offset = 35040           # The number of data points in a year.
    EWC_importance = 100000 
    EWC_sample_size = 1000
    GEM_memory_per_task = 250
    RPR_dict, gem, EWC_dict = None, None, None
    offset_event_based = 0          # The offset for the eALG mode, data points to use before event
    CL_method = False               # if yes: fine-tune the model from previous update step, if false => TL method or model from scratch

    eval_metrics = ["rmse", "mae", "mase"]

    ## load parameters from config
    adaptive_learning_config = config["adaptive_learning"]
    update_interval = adaptive_learning_config["update_interval"]
    train_val_interval = adaptive_learning_config["train_val_interval"]
    test_interval = adaptive_learning_config["test_interval"]
    output_file_name = adaptive_learning_config["output_file_name"]
    max_len = year_offset * adaptive_learning_config["number_of_years"]
    mode = adaptive_learning_config["mode"]
    print(f"Running in mode : {mode}")

    update_steps = 1 + int((max_len - (train_val_interval + test_interval)) / update_interval)

    general_TL_model = load_model()

    ## Pre-setup for different modes
    if mode in ["Scratch"]:         # scratch method
        CL_method = False
        general_TL_model = None
        # need at least 50 epochs to converge
        config["training"]["epochs"] = 50

    elif mode in ["ALG", "eALG", "GIL"]:        # TL methods
        CL_method = False

    elif mode in ["AL", "IL", "IFT", "SML"]:    # CL methods
        CL_method = True

    elif mode == "EWC":                         # CL method
        CL_method = True
        feature_scaler, target_scaler = load_scaler()

    elif mode == "GEM":                         # CL method
        CL_method = True
        gem = GEM(
            interval_count=update_steps,
            memory_per_task=GEM_memory_per_task,
            device=device
        )

    elif mode == "RPR":                         # CL method
        CL_method = True
        RPR_dict = {"reinit_prop": 0.003}

    # Get all paths from CSVs we want to do adaptive learning on.
    csv_paths = []
    schedules = {}

    for (root,_,files) in os.walk(config["data"]["folder"], topdown=True):

        for file in files:
            if not (file.endswith('.csv') and file.startswith("_")):
                continue

            path_to_csv = os.path.join(root, file)

            # read the drifting schedule of each target. This is needed to set the event-based collection time
            if mode == "eALG":
                schedule = pd.read_csv(os.path.join(root, "schedule.csv"))
                schedules[file] = [(schedule["timestamp"][i], schedule["eventtype"][i]) for i in range (len(schedule["timestamp"])) if schedule["eventtype"][i] in [1,2]]
            csv_paths.append(path_to_csv)

    errors = []

    for idx, csv_path in enumerate(csv_paths):

        csv_base_name = os.path.basename(csv_path)

        err_csv, EWC_old_data_list = [], []

        print(f"Currently at CSV {idx + 1} / {len(csv_paths)}. CSV Name: {csv_base_name}")

        for update_step in range(update_steps):

            config_ = deepcopy(config)

            config_["data"]["folder"] = csv_path

            # Looks the following:    UPDATE_INTERVAL * i | TRAIN_INTERVAL | TEST_INTERVAL | REST (not considered in this iteration)
            start_offset = update_interval * update_step
            end_train_interval = start_offset + train_val_interval - 1
            end_test_interval = end_train_interval + test_interval

            config_["data"]["dataframe_start"] = start_offset
            config_["data"]["dataframe_limit"] = end_train_interval

            # for Accumulative or learning from Scratch we always use the whole dataset
            if mode in ["AL", "Scratch", "ALG"]:
                config_["data"]["dataframe_start"] = 0

            elif mode == "GEM":
                gem.current_interval = update_step

            elif mode == "EWC":
                if update_step > 0:     
                    train_dataloader = preprocess_data_for_evaluation(
                        forecast_horizon=config_["model"]["forecast_horizon"],
                        scaler_features=feature_scaler,
                        scaler_target=target_scaler,
                        **config_["data"],
                        **config_["dataloader"]
                    )

                    sample_indices = random.sample(range(len(train_dataloader.dataset)), EWC_sample_size)
                    samples = [train_dataloader.dataset[i] for i in sample_indices]

                    EWC_old_data_list = EWC_old_data_list + samples
                    EWC_old_data_list = random.sample(EWC_old_data_list, k=EWC_sample_size)

                EWC_dict = {
                    "importance": EWC_importance,
                    "datasets": EWC_old_data_list
                }

            elif mode in ["eALG"]:
                # reversed ensures that the most recent event is selected.
                for event in reversed(schedules[os.path.basename(csv_path)]):
                    if start_offset > event[0]/900:
                        print(f"Model from Scratch or Accumulative Learning mode: using data from {int(event[0]/900 - offset_event_based)} onwards")
                        config_["data"]["dataframe_start"] = int(event[0]/900 - offset_event_based)
                        break
                # only executed if no recent event could be found (no break executed)
                else:
                    print("Model from Scratch or Accumulative Learning mode: using whole dataset")
                    config_["data"]["dataframe_start"] = 0

            elif mode == "SML":
                config_["data"]["dataframe_start"] = 0
                train_ranges = []
                val_ranges = []

                split_train = config_["data"].get("train_split", 0.7)

                if not train_ranges:
                    first_start = start_offset
                    first_end = first_start + int((end_train_interval - first_start) * split_train)
                    train_ranges = [[first_start, first_end]]
                if not val_ranges:
                    val_ranges = compute_val_ranges(train_ranges, start_offset, end_train_interval)

                # Add historical ranges from shifted train ranges
                historical_ranges = []
                for start, end in train_ranges:
                    old_start = start - year_offset
                    old_end = old_start + (2 * train_val_interval + test_interval)
                    years_back = 1                          # can be modified in case more then 1 year of seasonal information is desired
                    while old_start >= 0 and years_back > 0:
                        historical_ranges.append([old_start, old_end])
                        old_start -= year_offset
                        old_end -= year_offset
                        years_back -= 1

                full_train_ranges = train_ranges + historical_ranges

                print(f"Training ranges for SML (step {update_step}): {full_train_ranges}")
                print(f"Validation ranges for SML (step {update_step}): {val_ranges}")
                print(f"End train interval: {end_train_interval}, train valinterval {train_val_interval}")

                config_["data"]["train_ranges"] = full_train_ranges
                config_["data"]["val_ranges"] = val_ranges
                config_["data"]["train_ranges_col_name"] = "index"


            # Training / Fine-Tuning the pretrained model.
            # TL methods
            if (not CL_method) or (update_step == 0):     # always take the first model to update.
                _, model_prev = train(pretrained_model_dict=general_TL_model, config=config_, ewc_dict= EWC_dict, rpr_dict=RPR_dict, gem=gem)

            # CL methods
            elif mode in ["IL", "AL", "SML", "EWC", "GEM", "RPR"]:      # take the previous model and update it again.
                _, model_prev = train(pretrained_model_dict=model_prev, config=config_, ewc_dict= EWC_dict, rpr_dict=RPR_dict, gem=gem)
            
            # Evaluate / Test the model on the test set
            config_["data"]["dataframe_start"] = end_train_interval
            config_["data"]["dataframe_limit"] = end_test_interval - 1

            # Result will be a dict with the eval_metrics as keys and the corresponding errors as values.
            metric_result = evaluate_model(pretrained_model_dict=model_prev, config=config_, eval_metrics=eval_metrics)

            errors.append([idx, csv_base_name, update_step] + [v for _, v in metric_result.items()])
            err_csv.append([v for _, v in metric_result.items()])

        # Add aggregated errors at the end
        err_aggr = [sum(v) / len(v) for v in zip(*err_csv)]
        errors.append([idx, csv_base_name, "avg"] + err_aggr)

    # Print the errors into a csv
    cols = ["idx", "csv_base_name", "update_step"] + [metric for metric in eval_metrics]
    df = pd.DataFrame(errors, columns=cols)

    output_path = f"outputs/{output_file_name}"
    os.makedirs(output_path, exist_ok=True)
    print(f"Saving results to {output_path}.csv")

    df.to_csv(f"{output_path}/errors.csv", index=False)

    return output_path, update_steps


def update_config(config: dict, hyperparams: dict = {}):

    for k, v in hyperparams.items():
        for key, value in v.items():
            config[k][key] = value
    return config


def compute_val_ranges(train_ranges, start_offset, end_train_interval):
    val_ranges = []
    current = start_offset

    for start, end in train_ranges:
        if current < start:
            val_ranges.append([current, start - 1])
        current = end + 1

    if current <= end_train_interval:
        val_ranges.append([current, end_train_interval])

    return val_ranges