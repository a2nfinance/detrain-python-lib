from torch.distributed._tensor.device_mesh import init_device_mesh

from torch.distributed.tensor.parallel import (
    parallelize_module
)
import torch.distributed as dist
import torch
import os

# Initialize a tensor parallel model.
def get_tp_model(model, parallelize_plan, device_type, mesh_shape):
    # DeviceMesh is a higher level abstraction that manages ProcessGroup
    # Mode details, see https://pytorch.org/tutorials/recipes/distributed_device_mesh.html 
    # device_type: cuda or cpu
    device_mesh = init_device_mesh(device_type=device_type, mesh_shape=mesh_shape)
    model = model.to(device_type)
    # Tensor parrallel model: https://pytorch.org/docs/stable/distributed.tensor.parallel.html
    tp_model = parallelize_module(
        module=model,
        device_mesh=device_mesh,
        parallelize_plan=parallelize_plan,
    )
    return tp_model

def save_model(model, name):
    rank = int(os.environ["RANK"])
    # Ensure all distributed process are completed
    dist.barrier()
    states = model.state_dict()
    if rank == 0:
        torch.save(states, f"{name}.pt")