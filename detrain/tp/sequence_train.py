def sequence_train_loop(dataloader, tp_model, loss_fn, optimizer, batch_size, device, rank):
    size = len(dataloader.dataset)
    tp_model.train()
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = tp_model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        # Logs needed for node rank 0 only.
        if (rank == 0):
            if batch % 100 == 0:
                loss, current = loss.item(), batch * batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")