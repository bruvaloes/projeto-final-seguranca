import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import IMG_SIZE


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


if __name__ == "__main__":
    loader_treino, loader_teste = criar_loaders()
    print(f"treino: {len(loader_treino.dataset)} imagens | teste: {len(loader_teste.dataset)} imagens")
    x, y = next(iter(loader_treino))
    print(x.shape, x.min().item(), x.max().item(), y[:10].tolist())
