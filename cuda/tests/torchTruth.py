import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation

        self.fc_layers = nn.ModuleList()
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.fc_layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size

        self.output_layer = nn.Linear(prev_size, output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        for fc_layer in self.fc_layers:
            x = self.activation(fc_layer(x))
        x = self.output_layer(x)
        return x

# Example usage
input_size = 10
hidden_sizes = [20, 30]
output_size = 5

model = MLP(input_size, hidden_sizes, output_size, F.relu)
print(model)
print(model.forward(torch.randn(1, 10)))
