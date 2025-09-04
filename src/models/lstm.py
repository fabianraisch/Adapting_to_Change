from torch import nn

class LSTM(nn.Module):

    def __init__(self, 
                 num_features: int = 8, 
                 hidden_size: int = 8, 
                 num_layers:int = 1, 
                 forecast_horizon: int = 1, 
                 dropout: float = 0.0,
                 num_targets: int = 1,
                 **kwargs
                 ):

        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.num_targets = num_targets
        self.lstm = nn.LSTM(
            input_size=num_features, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout)
        
        self.fc = nn.Linear(
            hidden_size, 
            forecast_horizon * num_targets
            )
    

    def forward(self, x):

        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Get the last time step's output
        lstm_out = self.fc(lstm_out)  # Predict for all targets and forecast horizon
        lstm_out = lstm_out.view(-1, self.forecast_horizon, self.num_targets)  # Reshape to (batch_size, forecast_horizon, num_targets)
        return lstm_out