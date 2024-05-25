import torch
from torch.distributed._tensor import DTensor
from torch.distributed.tensor.parallel import loss_parallel
import torch.distributed as dist
def test_loop(dataloader, tp_model, loss_fn, device, rank):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    tp_model.eval()
    size = len(dataloader.dataset)
    tp_loss = torch.zeros(3).to(device)
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # A context manager that enables loss parallelism, 
            # where efficient parallelized loss computation can be performed when the input is sharded on the class dimension. 
            # Currently only the cross-entropy loss is supported.
            with loss_parallel():
                pred = tp_model(X)
                # Loss
                tp_loss[0] += loss_fn(pred, y).item()
                pred = DTensor.to_local(pred)

                # Correct
                tp_loss[1] += (pred.argmax(1) == y).type(torch.float).sum().item()

                # Batches
                tp_loss[2] += 1
        dist.all_reduce(tp_loss, op=dist.ReduceOp.SUM)
    if (rank == 0):
        test_loss = tp_loss[0] / tp_loss[2]
        correct = tp_loss[1] / size
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, int(tp_loss[1]), int(size),
            100. * correct))