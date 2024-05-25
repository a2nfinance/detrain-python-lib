import torch.distributed.rpc as rpc
from detrain.ppl.master_node import run_master
from detrain.ppl.dis_model import DistributedModel
from torch.distributed.optim.optimizer import DistributedOptimizer
# Each worker will be used for a model shard.
# rank: node rank
# world_size: total processes in the parallel training (number of nodes * number of processes per node)
def run_worker(rank, world_size, model_params, train_dataloader, test_dataloader, loss_fn, optimized_class, epochs, batch_size, lr):
    # Higher timeout is added to accommodate for kernel compilation time in case of ROCm.
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=300)
    
    # Rank 0: a master node
    # Rank > 0: worker nodes
    # The master node is used for tensor offloading
    # Worker nodes are used for training model shards.
    # All nodes and processes must be initialized using RPC, then the training process will start.
    if rank == 0:
        print("--- Init master RPC")
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        print("--- Done init master")

        # Create an instance of DistributedModel
        model = DistributedModel(
            # Split size
            model_params[0],
            # Workers
            model_params[1],
            # Node devices
            model_params[2],
            # Model shards
            model_params[3]
        )

        # Create a distributed optimizer based on a base optimizer python class.
        optimizer = DistributedOptimizer(
            optimized_class,
            model.parameter_rrefs(),
            lr=lr,
        )

        # Run a master node.
        # See masternode.py.
        run_master(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs, batch_size)
    else:
        print(f"--- Init worker {rank} RPC")
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        print(f"--- Start to listen & receive the forwarded data from the master node")
        pass

    # block until all rpcs finish
    rpc.shutdown()
