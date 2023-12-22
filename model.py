import torch.nn as nn

class DeepQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)