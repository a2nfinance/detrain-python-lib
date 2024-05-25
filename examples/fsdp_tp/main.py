import torch.nn as nn
import time
import os
import torch
from detrain.ppl.args_util import get_args
from detrain.ppl.dataset_util import get_torchvision_dataset
from detrain.fsdp_tp.train_eval import train_eval
from detrain.fsdp_tp.model_2d import get_model_2d

from base_model import NeuralNetwork

import torch.optim as optim
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed._tensor import Shard

if __name__=="__main__":
    # Get torchrun args
    args = get_args()
    world_size = int(os.environ["WORLD_SIZE"])
    node_rank = int(os.environ["RANK"])
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    device = "cpu"

    # Check devices
    if (args.gpu is not None):
        arr = args.gpu.split('_')
        for dv in range(len(arr)):
            if dv == node_rank:
                if int(arr[dv]) == 1:
                    device = "cuda"
    

    # Define optimizer & loss_fn
    loss_fn = nn.CrossEntropyLoss()

    # Model
    model = NeuralNetwork().to(device)
    tp_size = 2
    model_2d = get_model_2d(model, {
        "in_proj": ColwiseParallel(
            input_layouts=Shard(0),
        ),
        "linear1": RowwiseParallel(
        ),
        "out_proj": ColwiseParallel(
            output_layouts=Shard(0),
        ),
    } , device, tp_size)
    # Create an optimizer for the parallelized module.
    optimizer = optim.AdamW(model_2d.parameters(), lr=lr, foreach=True)
    
    # Dataloaders

    (train_dataloader, test_dataloader) = get_torchvision_dataset("MNIST", batch_size)

    tik = time.time()
    train_eval(
        model_2d, 
        train_dataloader, 
        test_dataloader, 
        loss_fn, 
        optimizer, 
        epochs, 
        batch_size,
        device
    )
    tok = time.time()
    print(f"Execution time = {tok - tik}")