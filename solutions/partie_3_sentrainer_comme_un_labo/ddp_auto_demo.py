
import os
import torch, torch.distributed as dist
import torch.nn as nn, torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
dist.init_process_group("gloo", rank=rank, world_size=world)

torch.manual_seed(42)
model = nn.Linear(4, 1, bias=False)
ddp_model = DDP(model)                 # aucune ligne d'all-reduce a ecrire
torch.manual_seed(100 + rank)
x, y = torch.randn(8, 4), torch.randn(8, 1)
F.mse_loss(ddp_model(x), y).backward() # DDP declenche l'all-reduce ICI

g = model.weight.grad
vals = [torch.zeros(1) for _ in range(world)] if rank == 0 else None
dist.gather(g[0, 0].reshape(1), vals, dst=0)
if rank == 0:
    for r in range(world):
        print(f"worker {r} | grad DDP = {vals[r].item():+.4f}")
dist.destroy_process_group()
