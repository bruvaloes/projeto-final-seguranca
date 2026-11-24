import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import IMG_SIZE, SinalNet


def criar_loaders(data_dir="data", batch_size=128, workers=2):
    # sem Normalize aqui: o modelo recebe pixels em [0, 1]
    tf_treino = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    tf_teste = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    treino = datasets.GTSRB(data_dir, split="train", transform=tf_treino, download=True)
    teste = datasets.GTSRB(data_dir, split="test", transform=tf_teste, download=True)

    pin = torch.cuda.is_available()
    loader_treino = DataLoader(treino, batch_size=batch_size, shuffle=True,
                               num_workers=workers, pin_memory=pin)
    loader_teste = DataLoader(teste, batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=pin)
    return loader_treino, loader_teste


def treinar_epoca(model, loader, criterio, otimizador, device):
    model.train()
    perda_total, acertos, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        otimizador.zero_grad()
        logits = model(x)
        perda = criterio(logits, y)
        perda.backward()
        otimizador.step()

        perda_total += perda.item() * x.size(0)
        acertos += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return perda_total / total, acertos / total


def main():
    epocas, batch_size, lr = 10, 128, 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"usando {device}")

    loader_treino, loader_teste = criar_loaders(batch_size=batch_size)
    model = SinalNet().to(device)
    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=epocas)

    for epoca in range(1, epocas + 1):
        inicio = time.time()
        perda, acc = treinar_epoca(model, loader_treino, criterio, otimizador, device)
        scheduler.step()
        print(f"epoca {epoca}/{epocas} | perda {perda:.4f} | acc treino {acc:.4f} | {time.time() - inicio:.0f}s")


if __name__ == "__main__":
    main()
