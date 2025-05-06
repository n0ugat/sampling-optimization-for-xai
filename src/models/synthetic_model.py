import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, input_size, hidden_layers=2, hidden_size=128, output_size=8):
        super(LinearModel, self).__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Create a list of layers
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_size, output_size))
        
        # Combine the list of layers into a sequential model
        self.model = nn.Sequential(*layers)
        
    def forward(self, x, only_feats=False, feats_and_class=False):
        x = self.model(x)
        return x.view(x.size(0), -1)