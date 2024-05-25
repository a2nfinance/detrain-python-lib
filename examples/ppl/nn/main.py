import torch.nn as nn
import torch
import time
import os
from detrain.ppl.args_util import get_args
from detrain.ppl.worker import run_worker
from detrain.ppl.dataset_util import get_torchvision_dataset
from shards_model import NNShard1, NNShard2
import torch.optim as optim

if __name__=="__main__":
    # Get torchrun args
    args = get_args()
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    lr = float(args.lr)

    # Check cuda device
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)

    # Define params to create a new instance of DistributedModel
    devices = []
    workers = []
    shards = [NNShard1, NNShard2]
    
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

    # Define optimizer & loss_fn
    loss_fn = nn.CrossEntropyLoss()
    optimizer_class = optim.SGD
    
    # Dataloaders

    (train_dataloader, test_dataloader) = get_torchvision_dataset("MNIST", batch_size)

    
    print(f"World_size: {world_size}, Rank: {rank}")
    
    # batch to mini batches
    num_split = 4
    tik = time.time()

    # Start a master node & worker nodes
    run_worker(
        rank, 
        world_size, 
        (
            args.split_size, 
            workers,
            devices, 
            shards
        ), 
        train_dataloader, 
        test_dataloader, 
        loss_fn, 
        optimizer_class, 
        epochs, 
        batch_size,
        lr
    )

    tok = time.time()
    print(f"number of splits = {num_split}, execution time = {tok - tik}")