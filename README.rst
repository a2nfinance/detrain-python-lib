==============
DeTrain
==============

Overview
--------

DeTrain is a Python package designed to train AI models using model parallelism methods. This package focuses on pipeline and tensor parallelism.

Installation
------------

You can install DeTrain using pip:

.. code-block:: sh

    pip install detrain

Usage
-----

Once installed, you can use DeTrain in your Python scripts like this:

.. code-block:: python

    import torch.nn as nn
    import torch
    import time
    import os
    from detrain.ppl.args_util import get_args
    from detrain.ppl.worker import run_worker
    from detrain.ppl.dataset_util import get_torchvision_dataset
    from shards_model import NNShard1, NNShard2
    import torch.optim as optim

    if __name__=="__main__":
        args = get_args()
        # Get args
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        epochs = int(args.epochs)
        batch_size = int(args.batch_size)
        lr = float(args.lr)

        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(i).name)

        devices = []
        workers = []
        shards = [NNShard1, NNShard2]
        # Check devices
        if (args.gpu is not None):
            arr = args.gpu.split('_')
            for dv in range(len(arr)):
                if dv > 0:
                    workers.append(f"worker{dv}")
                    if int(arr[dv]) == 1:
                        devices.append("cuda:0")
                    else:
                        devices.append("cpu")

        # Define optimizer & loss_fn
        loss_fn = nn.CrossEntropyLoss()
        optimizer_class = optim.SGD
        
        # Dataloaders

        (train_dataloader, test_dataloader) = get_torchvision_dataset("MNIST", batch_size)

        
        print(f"World_size: {world_size}, Rank: {rank}")
        
        num_split = 4
        tik = time.time()
        run_worker(
            rank, 
            world_size, 
            (
                args.split_size, 
                workers,
                devices, 
                shards
            ), 
            train_dataloader, 
            test_dataloader, 
            loss_fn, 
            optimizer_class, 
            epochs, 
            batch_size,
            lr
        )
        tok = time.time()
        print(f"number of splits = {num_split}, execution time = {tok - tik}")

For detailed examples, please visit the `DeTrain examples <https://github.com/a2nfinance/detrain-example>`_.

Contributing
------------

Contributions are welcome! If you'd like to contribute to DeTrain, please follow these steps:

1. Fork the repository on GitHub.
2. Create a new branch.
3. Make your changes and commit them with clear descriptions.
4. Push your changes to your fork.
5. Submit a pull request.

Bug Reports and Feedback
------------------------

If you encounter any bugs or have feedback, please open an issue on the GitHub repository.

License
-------

DeTrain is licensed under the MIT License. See the LICENSE file for more information.
