import torch.distributed as dist
from torch.distributed.elastic.rendezvous import c10d_rendezvous_backend
from torch.distributed.elastic.rendezvous import dynamic_rendezvous
store = dist.TCPStore("localhost", 9999)

backend = c10d_rendezvous_backend.C10dRendezvousBackend(store, "101")



if __name__=="__main__":
    
    rdzv_handler = dynamic_rendezvous.DynamicRendezvousHandler.from_backend(
        run_id="101",
        store=store,
        backend=backend,
        min_nodes=2,
        max_nodes=4                                                                             
    )
