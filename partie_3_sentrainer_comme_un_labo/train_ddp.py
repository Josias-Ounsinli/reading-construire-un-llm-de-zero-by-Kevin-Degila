"""train_ddp.py : entraîner le petit GPT des fables en DDP (parallélisme de données distribué).

Le MEME script tourne :
  - hors ligne, sur ton CPU, avec 2 processus et le backend gloo (pour comprendre),
  - sur un noeud cloud, avec N GPU, avec le backend nccl (pour la vraie chose).

Il détecte tout seul cuda -> nccl / cpu -> gloo. Tu ne changes pas une ligne
entre ta machine et le noeud loué.

Lancement (hors ligne, 2 workers CPU) :
    torchrun --nproc_per_node=2 --nnodes=1 train_ddp.py --steps 200

Lancement (noeud cloud, 8 GPU) :
    torchrun --nproc_per_node=8 --nnodes=1 train_ddp.py --steps 2000

torchrun fixe pour nous les variables d'environnement RANK, WORLD_SIZE et
LOCAL_RANK, et lance nproc_per_node copies de ce fichier. Chaque copie est un
"worker" : un processus qui tient une replique complete du modele.
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


# --- Le corpus : un extrait des fables du chapitre 1, embarque dans le fichier. ---
CORPUS = (
    "LE CORBEAU ET LE RENARD\n"
    "Maitre corbeau, sur un arbre perche,\n"
    "Tenait en son bec un fromage.\n"
    "Maitre renard, par l'odeur alleche,\n"
    "Lui tint a peu pres ce langage :\n"
    "Et bonjour, Monsieur du Corbeau.\n"
    "Que vous etes joli ! que vous me semblez beau !\n"
    "Sans mentir, si votre ramage\n"
    "Se rapporte a votre plumage,\n"
    "Vous etes le phenix des hotes de ces bois.\n"
    "A ces mots le corbeau ne se sent pas de joie ;\n"
    "Et pour montrer sa belle voix,\n"
    "Il ouvre un large bec, laisse tomber sa proie.\n"
    "Le renard s'en saisit, et dit : Mon bon Monsieur,\n"
    "Apprenez que tout flatteur\n"
    "Vit aux depens de celui qui l'ecoute.\n"
    "Cette lecon vaut bien un fromage, sans doute.\n"
    "Le corbeau, honteux et confus,\n"
    "Jura, mais un peu tard, qu'on ne l'y prendrait plus.\n"
) * 6


# --- Le vocabulaire caractere, comme au chapitre 1. ---
CHARS = sorted(set(CORPUS))
STOI = {c: i for i, c in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS)
BLOCK_SIZE = 32


class FablesDataset(Dataset):
    """Chaque exemple = une fenetre de BLOCK_SIZE caracteres et sa cible decalee d'un cran."""

    def __init__(self, text):
        self.data = torch.tensor([STOI[c] for c in text], dtype=torch.long)

    def __len__(self):
        return len(self.data) - BLOCK_SIZE - 1

    def __getitem__(self, i):
        x = self.data[i : i + BLOCK_SIZE]
        y = self.data[i + 1 : i + BLOCK_SIZE + 1]
        return x, y


class MiniGPT(nn.Module):
    """Le petit GPT du chapitre 10, en version compacte (1 bloc), pour tenir sur un CPU."""

    def __init__(self, d_model=64, n_heads=4, d_ff=256):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos = nn.Embedding(BLOCK_SIZE, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB_SIZE, bias=False)
        self.register_buffer("mask", torch.triu(torch.ones(BLOCK_SIZE, BLOCK_SIZE), diagonal=1).bool())

    def forward(self, x):
        T = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        a, _ = self.attn(self.ln1(h), self.ln1(h), self.ln1(h), attn_mask=self.mask[:T, :T])
        h = h + a
        h = h + self.ffn(self.ln2(h))
        return self.head(self.ln_f(h))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()

    # 1. Qui suis-je dans la grappe ? torchrun a rempli ces variables.
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 2. Choisir le backend : nccl si des GPU sont la, gloo sinon (CPU).
    if torch.cuda.is_available():
        backend, device = "nccl", torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        backend, device = "gloo", torch.device("cpu")

    # 3. Ouvrir la ligne telephonique entre les workers.
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    is_main = rank == 0
    if is_main:
        print(f"[rank 0] backend={backend} world_size={world_size} device={device}", flush=True)

    torch.manual_seed(42)  # meme init sur tous les workers : replique identique

    # 4. Les donnees, coupees en world_size parts sans recouvrement.
    dataset = FablesDataset(CORPUS)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    # 5. Le modele, replique sur chaque worker, enveloppe dans DDP.
    model = MiniGPT().to(device)
    model = DDP(model, device_ids=[local_rank] if backend == "nccl" else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 6. La boucle d'entrainement : identique au chapitre 10, un all-reduce en plus (invisible).
    step = 0
    t0 = time.time()
    while step < args.steps:
        sampler.set_epoch(step)  # re-melange a chaque passage
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()          # DDP declenche l'all-reduce des gradients ICI
            optimizer.step()
            step += 1
            if is_main and step % 50 == 0:
                print(f"[rank 0] step {step:4d} | loss {loss.item():.3f}", flush=True)
            if step >= args.steps:
                break

    if is_main:
        print(f"[rank 0] termine en {time.time() - t0:.1f} s", flush=True)
        # Seul le rank 0 sauvegarde : les repliques sont identiques, un checkpoint suffit.
        torch.save(model.module.state_dict(), "gpt_ddp.pt")
        print("[rank 0] checkpoint ecrit : gpt_ddp.pt", flush=True)

    dist.destroy_process_group()  # raccrocher proprement


if __name__ == "__main__":
    main()
