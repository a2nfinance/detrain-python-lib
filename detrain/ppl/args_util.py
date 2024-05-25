import argparse
def get_args():
    parser = argparse.ArgumentParser(description='simple distributed training job based on pipeline parallelism')
    ## General settings
    parser.add_argument('--epochs', default=3, type=int, help='Total epochs to train the model')
    parser.add_argument('--batch_size', default=10, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument("--split_size", default=2, type=int, help="Number of parts of the input data, a batch will be device to mini batches")
    parser.add_argument('--log', default=None, type=int, 
                        help='Enable training logs, this action can affect to training process performance')
    ## Optimizer settings
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')

    ## Cluster settings
    parser.add_argument('--gpu', default=None, type=str, help="E.g. 0_1_1: the first node uses CPU, remain nodes use GPU")

    ## Model name
    parser.add_argument("--model_name", default="trained_model", type=str, help='Name of the trained model to be saved')
    args = parser.parse_args()
    return args
