from torch.distributed.device_mesh import init_device_mesh
import os

# DeviceMesh is a higher level abstraction that manages ProcessGroup
# Mode details, see https://pytorch.org/tutorials/recipes/distributed_device_mesh.html 
# device_type: cuda or cpu
def get_2d_mesh(device_type, tp_size):
    # Get from torchrun params
    _world_size = int(os.environ["WORLD_SIZE"])

    # Word_size = tp_size * dp_size, all numbers are integer.
    assert (
        _world_size % tp_size == 0
    ), f"World size {_world_size} needs to be divisible by TP size {tp_size}"

    dp_size = _world_size // tp_size

    # Init device mesh with two mesh names are DP, TP (DataParallelism, TensorParallelism)
    device_mesh = init_device_mesh(device_type,mesh_shape=(dp_size, tp_size), mesh_dim_names=("dp", "tp"))
    dp_mesh = device_mesh["dp"]
    tp_mesh = device_mesh["tp"]

    return (dp_mesh, tp_mesh)