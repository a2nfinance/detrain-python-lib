import torch
from detrain.ppl.train import train_loop
from detrain.ppl.evaluation import test_loop
from detrain.ppl.args_util import get_args

def run_master(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs, batch_size):
    args = get_args()
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, batch_size)
        test_loop(test_dataloader, model, loss_fn)
    # Save model
    states = model.state_dict()
    torch.save(states, f"{args.model_name}.pt")
    print("Done!")

