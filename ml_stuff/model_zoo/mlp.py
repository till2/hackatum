import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, ae, input_dim, hidden_dim, hidden_layers, output_dim, activation="ReLU"):
        super(MLP, self).__init__()
        fc_layers = []
        fc_layers.append(
            nn.ModuleDict(
                {
                    "lin": nn.Linear(input_dim, hidden_dim),
                    "bn": nn.BatchNorm1d(hidden_dim),
                }
            )
        )
        for _ in range(hidden_layers):
            fc_layers.append(
                nn.ModuleDict(
                    {
                        "lin": nn.Linear(hidden_dim, hidden_dim),
                        "bn": nn.BatchNorm1d(hidden_dim),
                    }
                )
            )
        fc_layers.append(nn.Linear(hidden_dim, output_dim))
        self.fc_layers = nn.ModuleList(fc_layers)
        self.activation = activation
        self.ae = ae
        self.ae.eval()

    def forward(self, x):
        x = self.ae.encode(x)
        for layer in self.fc_layers[:-1]:
            x = layer["lin"](x)
            if self.activation == "ReLU":
                x = nn.functional.elu(x)
            elif self.activation == "Sigmoid":
                x = nn.functional.sigmoid(x)
            else:
                raise ValueError("choose ReLU or Sigmoid as activation")
            x = layer["bn"](x)
        x = self.fc_layers[-1](x)
        x = nn.functional.sigmoid(x)
        return x
