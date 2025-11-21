# model.py
import torch
import torch.nn as nn

class Model(nn.Module):
    
    def __init__(
        self,
        weight_path: str | None,
        description: str,
        input_size: int,
        hidden_sizes: list[int],
        hidden_activation,
        output_activation,
        dropout_p: float = 0.5,
    ) -> None:
        super(Model, self).__init__()

        self.input_size = input_size

        layers: list[nn.Module] = []
        
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        for i in range(len(hidden_sizes) - 1):
            in_features = hidden_sizes[i]
            out_features = hidden_sizes[i + 1]
            is_last = (i == len(hidden_sizes) - 2)

            layers.append(nn.Linear(in_features, out_features))

            if is_last:
                layers.append(output_activation())
            else:
                layers.append(hidden_activation())
                if dropout_p is not None and dropout_p > 0:
                    layers.append(nn.Dropout(p=dropout_p))

        self.inner = nn.ModuleList(layers)
        self.description = description
        self.weight_path = weight_path
        self.description = description
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dropout_p = dropout_p

        if weight_path:
            self.load_state_dict(torch.load(weight_path))
            
    def reset_parameters(self) -> None:
        for layer in self.inner:
            layer.reset_para
        
    def forward(self, x):
        for layer in self.inner:
            x = layer(x)
        x = x.squeeze()
        return x

