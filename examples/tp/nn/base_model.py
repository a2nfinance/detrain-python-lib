import torch.nn as nn
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.in_proj = nn.Linear(28 * 28, 512)
        self.relu1 = nn.ReLU()
        self.linear1 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.out_proj = nn.Linear(512, 10)
        

    def forward(self, x):
        x = self.flatten(x)
        x = self.in_proj(x)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        logits = self.out_proj(x)
        return logits