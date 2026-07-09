
import os
import torch, torch.distributed as dist
import torch.nn as nn, torch.nn.functional as F

rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
dist.init_process_group("gloo", rank=rank, world_size=world)

torch.manual_seed(42)                 # MEME init -> repliques identiques
model = nn.Linear(4, 1, bias=False)
torch.manual_seed(100 + rank)         # donnees PROPRES a chaque worker
x, y = torch.randn(8, 4), torch.randn(8, 1)
F.mse_loss(model(x), y).backward()
g = model.weight.grad

before = [torch.zeros(1) for _ in range(world)] if rank == 0 else None
dist.gather(g[0, 0].reshape(1), before, dst=0)
dist.all_reduce(g, op=dist.ReduceOp.SUM)   # chaque worker recoit la SOMME
g /= world                                  # ... transformee en MOYENNE
after = [torch.zeros(1) for _ in range(world)] if rank == 0 else None
dist.gather(g[0, 0].reshape(1), after, dst=0)

if rank == 0:
    for r in range(world):
        print(f"worker {r} | AVANT all-reduce : grad = {before[r].item():+.4f}")
    moy = sum(b.item() for b in before) / world
    print(f"moyenne des gradients = {moy:+.4f}")
    for r in range(world):
        print(f"worker {r} | APRES all-reduce : grad = {after[r].item():+.4f}")
dist.destroy_process_group()
