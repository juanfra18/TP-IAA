# model.py
import torch
import torch.nn as nn

class Model(nn.Module):
    
    def __init__(
        self,
        weight_path: str | None,
        description: str,
        hidden_sizes: list[int],
        hidden_activation = nn.ReLU,
        output_activation = nn.Sigmoid,
        dropout_p: float = 0.5,   # <-- dropout probability
    ) -> None:
        super(Model, self).__init__()

        layers: list[nn.Module] = []

        for i in range(len(hidden_sizes) - 1):
            in_features = hidden_sizes[i]
            out_features = hidden_sizes[i + 1]
            is_last = (i == len(hidden_sizes) - 2)

            # Linear layer
            layers.append(nn.Linear(in_features, out_features))

            if is_last:
                # Output activation only on last layer
                layers.append(output_activation())
            else:
                # Hidden activation + dropout on hidden layers
                layers.append(hidden_activation())
                if dropout_p is not None and dropout_p > 0:
                    layers.append(nn.Dropout(p=dropout_p))

        self.inner = nn.ModuleList(layers)
        self.description = description

        if weight_path:
            self.load_state_dict(torch.load(weight_path))
        
    def forward(self, x):
        for layer in self.inner:
            x = layer(x)
        x = x.squeeze()
        return x

