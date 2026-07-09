
import os
import torch, torch.distributed as dist
import torch.nn as nn, torch.nn.functional as F

def run(sync):
    torch.manual_seed(42)
    model = nn.Linear(4, 1, bias=False)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(100 + rank)
    batches = [(torch.randn(8, 4), torch.randn(8, 1)) for _ in range(5)]
    for x, y in batches:
        F.mse_loss(model(x), y).backward()
        if sync:
            g = model.weight.grad
            dist.all_reduce(g, op=dist.ReduceOp.SUM); g /= world
        opt.step(); opt.zero_grad()
    return model.weight.detach()[0, 0].item()

rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
dist.init_process_group("gloo", rank=rank, world_size=world)
w_sync = run(sync=True)
w_nosync = run(sync=False)

s = [torch.zeros(1) for _ in range(world)] if rank == 0 else None
n = [torch.zeros(1) for _ in range(world)] if rank == 0 else None
dist.gather(torch.tensor([w_sync]), s, dst=0)
dist.gather(torch.tensor([w_nosync]), n, dst=0)
if rank == 0:
    for r in range(world):
        print(f"worker {r} | AVEC synchro : poids final = {s[r].item():+.4f}")
    for r in range(world):
        print(f"worker {r} | SANS synchro : poids final = {n[r].item():+.4f}")
dist.destroy_process_group()
