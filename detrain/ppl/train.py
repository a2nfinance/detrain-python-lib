import torch.distributed.autograd as dist_autograd
# dataloader: training dataloading
# model: an instance of DistributedModel class
# loss_fn: loss function
# optimizer: optimizer class
# batch_size: the number of data items in a batch.
def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # The distributed autograd context is the dedicated scope for the
        # distributed backward pass to store gradients, which can later be
        # retrieved using the context_id by the distributed optimizer.
        # See details here: https://pytorch.org/docs/stable/rpc/distributed_autograd.html
        with dist_autograd.context() as context_id:
            pred = model(X)
            loss = loss_fn(pred, y)
            dist_autograd.backward(context_id, [loss])
            optimizer.step(context_id)
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")