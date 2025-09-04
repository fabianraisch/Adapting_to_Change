import torch.nn as nn

class NaiveModel(nn.Module):
    '''
    Naive Model that just returns the last seen target variable
    '''
    def __init__(self, forecast_horizon=1, num_target_cols=1):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.num_targets = num_target_cols
        
        # Dummy layer just to make it compatible with some pipelines
        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        # Get the last timestep: [batch_size, num_features]
        last_step = x[:, -1, :]  # [B, F]

        # Assume the target cols are the last N columns of the input
        last_targets = last_step[:, -self.num_targets:]  # [B, num_target_cols]

        # Repeat over forecast horizon
        out = last_targets.unsqueeze(1).repeat(1, self.forecast_horizon, 1)  # [B, H, T]
        return out