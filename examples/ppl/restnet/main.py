import torch.nn as nn
import time
import torch
import os
from detrain.ppl.args_util import get_args
from detrain.ppl.worker import run_worker
from detrain.ppl.dis_model import DistributedModel
from detrain.ppl.dataset_util import get_torchvision_dataset
from shards_model import ResNetShard1, ResNetShard2
from torch.distributed.optim.optimizer import DistributedOptimizer
import torch.optim as optim
# Define model here

if __name__=="__main__":
    # Get torchrun args
    args = get_args()
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)


    # Check cuda device
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)

    # Define params to create a new instance of DistributedModel
    devices = []
    workers = []
    shards = [ResNetShard1, ResNetShard2]
    
    # Devices for model shards
    if (args.gpu is not None):
        arr = args.gpu.split('_')
        for dv in range(len(arr)):
            if dv > 0:
                workers.append(f"worker{dv}")
                if int(arr[dv]) == 1:
                    devices.append("cuda:0")
                else:
                    devices.append("cpu")

    model = DistributedModel(
        args.split_size, 
        workers,
        devices,
        shards
    )
    
    # Define optimizer & loss_fn
    loss_fn = nn.MSELoss()
    optimizer = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=args.lr,
    )
    # Dataloaders

    (train_dataloader, test_dataloader) = get_torchvision_dataset("M", batch_size)

    
    print(f"World_size: {world_size}, Rank: {rank}")
    num_split = 4
    tik = time.time()
    run_worker(rank, world_size, model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs, batch_size)
    tok = time.time()
    print(f"number of splits = {num_split}, execution time = {tok - tik}")