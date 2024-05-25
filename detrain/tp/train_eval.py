from detrain.tp.train import train_loop
from detrain.tp.evaluation import test_loop
from torch.distributed.tensor.parallel import loss_parallel
import os
def train_eval(tp_model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs, batch_size, device):
    rank = int(os.environ["RANK"])
    for t in range(epochs):
        print(f"\nRank{rank} -- Epoch {t+1} start\n-------------------------------")
        train_loop(train_dataloader, tp_model, loss_fn, optimizer, batch_size, device, rank)
        test_loop(test_dataloader, tp_model, loss_fn, device, rank)
    print("Done!")