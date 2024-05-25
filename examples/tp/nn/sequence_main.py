import torch
import torch.nn as nn
import time
import os
from detrain.ppl.args_util import get_args
from detrain.tp.sequence_train_eval import sequence_train_eval
from detrain.tp.model_utils import get_tp_model
from detrain.ppl.dataset_util import get_torchvision_dataset
import torch.optim as optim
from base_model import NeuralNetwork
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed._tensor import Shard

if __name__=="__main__":
    # Get torchrun args
    args = get_args()
    world_size = int(os.environ["WORLD_SIZE"])
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    device = "cpu"
    work_rank = torch.distributed.get_rank()

    # Check devices
    if (args.gpu is not None):
        arr = args.gpu.split('_')
        for dv in range(len(arr)):
            if dv == work_rank:
                if int(arr[dv]) == 1:
                    device = "cuda"
    
    # Define optimizer & loss_fn
    loss_fn = nn.CrossEntropyLoss()
    optimizer_class = optim.SGD
    model = NeuralNetwork().to(device)

    # Create 1D device mesh
    mesh_shape = (world_size, )

    # Return a Parallelize_modul
    # Input or output layouts are based on dim 0
    sp_model = get_tp_model(model, {
        "in_proj": ColwiseParallel(
            input_layouts=Shard(0),
        ),
        "linear1": RowwiseParallel(
            
        ),
        "out_proj": ColwiseParallel(
            output_layouts=Shard(0),
        ),
    } , device, mesh_shape)

    # Create an optimizer for the parallelized module.
    optimizer = torch.optim.AdamW(sp_model.parameters(), lr=lr, foreach=True)
    
    # Training & testing dataloader

    (train_dataloader, test_dataloader) = get_torchvision_dataset("MNIST", batch_size)

    tik = time.time()
    sequence_train_eval(
        sp_model, 
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