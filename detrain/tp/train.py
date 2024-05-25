from torch.distributed.tensor.parallel import loss_parallel

def train_loop(dataloader, tp_model, loss_fn, optimizer, batch_size, device, rank):
    tp_model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        
        X, y = X.to(device), y.to(device)
        pred = tp_model(X)
        # A context manager that enables loss parallelism, 
        # where efficient parallelized loss computation can be performed when the input is sharded on the class dimension. 
        # Currently only the cross-entropy loss is supported. 
        # More details see here
        # https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.loss_parallel
        with loss_parallel():
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        if (rank == 0):
            if batch % 100 == 0:
                loss, current = loss.item(), batch * batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")