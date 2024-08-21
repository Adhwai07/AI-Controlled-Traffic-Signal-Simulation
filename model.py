
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TrainModel(nn.Module):

    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim):
        super(TrainModel, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._num_layers = num_layers
        self._width = width
        

        self.linear1 = nn.Linear(self._input_dim, self._width)
        self.hidden_layer = []
        for _ in range(self._num_layers):
            self.hidden_layer.append(nn.Linear(self._width, self._width))
        self.linearLast = nn.Linear(self._width, self._output_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=self._learning_rate)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        for _ in range(self._num_layers):
            x = F.relu(self.hidden_layer[_](x)) 
        actions = self.linearLast(x)
        return actions


class TestModel:
    pass