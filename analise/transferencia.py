import argparse

import torch

from ataques.attack import pgd
from sinalnet.model import escolher_device
from sinalnet.predict import carregar_modelo
from sinalnet.train import criar_loaders


@torch.no_grad()
def _acc(model, x, y):
    return (model(x).argmax(1) == y).sum().item()


def avaliar_transferencia(origem, destino, loader, device, eps, alpha, passos, limite_amostras):
    origem.eval()
    destino.eval()
    acertos_origem, acertos_destino, total = 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = pgd(origem, x, y, eps, alpha, passos)
        acertos_origem += _acc(origem, x_adv, y)
        acertos_destino += _acc(destino, x_adv, y)
        total += x.size(0)
        if total >= limite_amostras:
            break

    return acertos_origem / total, acertos_destino / total


def main():
    parser = argparse.ArgumentParser(
        description="testa se exemplos adversariais gerados num modelo enganam o outro (transferencia)")
    parser.add_argument("--normal", default="sinalnet.pt")
    parser.add_argument("--robusto", default="sinalnet_robusto.pt")
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--passos", type=int, default=10)
    parser.add_argument("--amostra", type=int, default=512)
    parser.add_argument("--data", default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    device = escolher_device()
    print(f"usando {device}")
    _, loader_teste = criar_loaders(args.data, args.batch_size, args.workers)

    normal = carregar_modelo(args.normal)
    robusto = carregar_modelo(args.robusto)

    print(f"\ngerando adversariais (PGD, eps={args.eps}) a partir do modelo NORMAL:")
    acc_normal, acc_robusto = avaliar_transferencia(
        normal, robusto, loader_teste, device, args.eps, args.alpha, args.passos, args.amostra)
    print(f"  no proprio modelo (normal):  {acc_normal:.4f}")
    print(f"  transferido pro robusto:     {acc_robusto:.4f}")

    print(f"\ngerando adversariais (PGD, eps={args.eps}) a partir do modelo ROBUSTO:")
    acc_robusto2, acc_normal2 = avaliar_transferencia(
        robusto, normal, loader_teste, device, args.eps, args.alpha, args.passos, args.amostra)
    print(f"  no proprio modelo (robusto): {acc_robusto2:.4f}")
    print(f"  transferido pro normal:      {acc_normal2:.4f}")


if __name__ == "__main__":
    main()
