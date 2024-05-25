from detrain.fsdp_tp.mesh_utils import get_2d_mesh
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# This 2D model includes a Tensor parallel model and a Full sharded data parallel model.
# Mode details:
# Tensor parrallel: https://pytorch.org/docs/stable/distributed.tensor.parallel.html
# FSDP example: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
def get_model_2d(model, tp_plan, device_type, tp_size):
    # create two meshes for TP and DP.
    (dp_mesh, tp_mesh) = get_2d_mesh(device_type, tp_size)
    model_tp = parallelize_module(model, tp_mesh, tp_plan)
    model_2d = FSDP(model_tp, device_mesh=dp_mesh, use_orig_params=False, device_id=device_type)
    return model_2d