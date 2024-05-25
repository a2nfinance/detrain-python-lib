import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef

# A base class for building a distributed model for pipeline parallelism training jobs.
# This class helps model shards to be trained on distributed nodes.
# All communication tasks are handled and controlled using RPC and remote call functions.
# RRef: Remote references
# Workers: distributed nodes for training model shards 
class DistributedModel(nn.Module):
    """
    Assemble multi parts as an nn.Module and define pipelining logic
    """
    def __init__(self, split_size, workers, devices, model_shards, *args, **kwargs):
        super(DistributedModel, self).__init__()

        assert len(workers) == len(devices) and len(workers) == len(model_shards)

        self.split_size = split_size
        self.rrefs = []

        # Create remote references
        # Devices[w]: the device is used on a worker, it can be a CPU or GPU.
        for w in range(len(workers)):
            rref = rpc.remote(
                workers[w],
                model_shards[w],
                args = (devices[w],) + args,
                **kwargs
            )
            self.rrefs.append(rref)
            
    def forward(self, xs):
        # Split the input batch xs into micro-batches, and collect async RPC
        # futures into a list
        out_futures = []
        for x in iter(xs.split(self.split_size, dim=0)):
            x_rref = RRef(x)
            # Forward input data to the first model shard on the first worker node.
            y_rref = self.rrefs[0].remote().forward(x_rref)
            for i in range(len(self.rrefs)):
                if i != 0:
                    # Forward output to the next model shard on the next worker node.
                    # This output is sent asynchronously.      
                    y_rref = self.rrefs[i].rpc_async().forward(y_rref)
                    out_futures.append(y_rref)

        # Wait for all asynchronous calls to complete.
        # Collect and concatenate all output tensors into one tensor.
        return torch.cat(torch.futures.wait_all(out_futures))

    def parameter_rrefs(self):
        remote_params = []
        # Collect all remote reference parameters into one array.
        for i in range(len(self.rrefs)):
            remote_params.extend(self.rrefs[i].remote().parameter_rrefs().to_here())
        return remote_params
