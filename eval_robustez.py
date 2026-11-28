import argparse
import csv

import torch

from ataques.attack import fgsm, pgd
from model import escolher_device
from predict import carregar_modelo
from train import criar_loaders


def avaliar_sweep(model, loader, device, metodo, eps_lista, alpha, passos, limite_amostras=None):
    model.eval()
    contagem = {eps: [0, 0] for eps in eps_lista}
    vistas = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        for eps in eps_lista:
            if eps == 0:
                x_pert = x
            elif metodo == "fgsm":
                x_pert = fgsm(model, x, y, eps)
            else:
                x_pert = pgd(model, x, y, eps, alpha, passos)
            with torch.no_grad():
                acertos = (model(x_pert).argmax(1) == y).sum().item()
            contagem[eps][0] += acertos
            contagem[eps][1] += x.size(0)

        vistas += x.size(0)
        if limite_amostras is not None and vistas >= limite_amostras:
            break

    return {eps: c / t for eps, (c, t) in contagem.items()}


def main():
    parser = argparse.ArgumentParser(
        description="curva de acuracia vs forca do ataque (FGSM/PGD), no conjunto de teste do GTSRB")
    parser.add_argument("--modelo", nargs="+", default=["sinalnet.pt"],
                         help="um ou mais checkpoints para comparar")
    parser.add_argument("--metodo", choices=["fgsm", "pgd"], default="pgd")
    parser.add_argument("--eps", nargs="+", type=float, default=[0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1])
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--passos", type=int, default=10)
    parser.add_argument("--data", default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--amostra", type=int, default=None,
                         help="limita a quantas imagens do teste avaliar (mais rapido; omitir usa as 12630 todas)")
    parser.add_argument("--csv", default=None, help="salva os resultados nesse arquivo CSV")
    args = parser.parse_args()

    device = escolher_device()
    print(f"usando {device}")
    _, loader_teste = criar_loaders(args.data, args.batch_size, args.workers)

    linhas = []
    for caminho in args.modelo:
        model = carregar_modelo(caminho)
        resultado = avaliar_sweep(model, loader_teste, device, args.metodo,
                                   args.eps, args.alpha, args.passos, args.amostra)
        print(f"\n{caminho} ({args.metodo}):")
        for eps in args.eps:
            print(f"  eps={eps:.3f}  acuracia={resultado[eps]:.4f}")
            linhas.append({"modelo": caminho, "metodo": args.metodo, "eps": eps, "acuracia": resultado[eps]})

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            escritor = csv.DictWriter(f, fieldnames=["modelo", "metodo", "eps", "acuracia"])
            escritor.writeheader()
            escritor.writerows(linhas)
        print(f"\nresultados salvos em {args.csv}")


if __name__ == "__main__":
    main()
