from detrain.tp.sequence_train import sequence_train_loop
def train_loop(dataloader, model_2d, loss_fn, optimizer, batch_size, device):
    sequence_train_loop(dataloader, model_2d, loss_fn, optimizer, batch_size, device)