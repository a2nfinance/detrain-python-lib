import torch.nn as nn
import threading
from torch.distributed.rpc import RRef

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self.flatten = nn.Flatten()

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]