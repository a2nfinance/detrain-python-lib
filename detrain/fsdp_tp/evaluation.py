import torch
from torch.distributed.tensor.parallel import loss_parallel
import torch.distributed as dist
def test_loop(dataloader, model_2d, loss_fn, device, rank):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model_2d.eval()
    model_2d_loss = torch.zeros(4).to(device)
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # A context manager that enables loss parallelism, 
            # where efficient parallelized loss computation can be performed when the input is sharded on the class dimension. 
            # Currently only the cross-entropy loss is supported.
            with loss_parallel():
                pred = model_2d(X)
                # Loss
                model_2d_loss[0] += loss_fn(pred, y).item()
                # Correct
                model_2d_loss[1] += (pred.argmax(1) == y).type(torch.float).sum().item()
                # Batches
                model_2d_loss[2] += 1
                # Size
                model_2d_loss[3] += len(X)
        dist.all_reduce(model_2d_loss, op=dist.ReduceOp.SUM)
    if (rank == 0):
        test_loss = model_2d_loss[0] / model_2d_loss[2]
        correct = model_2d_loss[1] / model_2d_loss[3]
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, int(model_2d_loss[1]), int(model_2d_loss[3]),
            100. * correct))