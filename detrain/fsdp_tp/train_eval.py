from detrain.fsdp_tp.evaluation import test_loop
from detrain.tp.sequence_train import sequence_train_loop
import os
def train_eval(model_2d, train_dataloader, test_dataloader, loss_fn, optimizer, epochs, batch_size, device):
    rank = int(os.environ["RANK"])
    for t in range(epochs):
        print(f"\nRank{rank} -- Epoch {t+1} start\n-------------------------------")
        sequence_train_loop(train_dataloader, model_2d, loss_fn, optimizer, batch_size, device, rank)
        test_loop(test_dataloader, model_2d, loss_fn, device, rank)
    print("Done!")