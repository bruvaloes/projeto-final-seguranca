import argparse
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from attack import pgd
from augmentacao import ruido_gaussiano
from model import CLASSES, IMG_SIZE, NUM_CLASSES, SinalNet, escolher_device


def criar_loaders(data_dir="data", batch_size=128, workers=2):
    # Augmentation pensado para fotos e frames de video reais: placa torta,
    # fora do centro, vista de lado, com luz diferente ou borrada, ruidosa
    # ou parcialmente coberta (sujeira, adesivo, sombra).
    # Sem flip horizontal: espelhar "vire a esquerda" vira "vire a direita".
    # Sem Normalize aqui: o modelo recebe pixels em [0, 1].
    tf_treino = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.3),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.3),
        transforms.ToTensor(),
        transforms.Lambda(ruido_gaussiano),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), value="random"),
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


def treinar_epoca(model, loader, criterio, otimizador, device, adv=None):
    model.train()
    perda_total, acertos, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if adv is not None and adv["eps"] > 0:
            # so uma fracao do batch fica adversarial (o resto continua limpo),
            # para nao sacrificar tanta acuracia limpa em troca de robustez
            k = int(x.size(0) * adv["proporcao"])
            if k > 0:
                # gera os exemplos adversariais com BatchNorm em modo eval, para
                # as iteracoes do PGD nao sujarem as estatisticas de rodagem
                model.eval()
                x_adv = pgd(model, x[:k], y[:k], adv["eps"], adv["alpha"], adv["passos"])
                model.train()
                x = torch.cat([x_adv, x[k:]], dim=0)

        otimizador.zero_grad()
        logits = model(x)
        perda = criterio(logits, y)
        perda.backward()
        otimizador.step()

        perda_total += perda.item() * x.size(0)
        acertos += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return perda_total / total, acertos / total


@torch.no_grad()
def avaliar(model, loader, device):
    model.eval()
    acertos, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        acertos += (model(x).argmax(1) == y).sum().item()
        total += x.size(0)
    return acertos / total


def avaliar_robustez(model, loader, device, eps=0.03, alpha=0.01, passos=10):
    # acuracia contra um atacante PGD (nao-direcionado) no proprio rotulo
    model.eval()
    acertos, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = pgd(model, x, y, eps, alpha, passos)
        with torch.no_grad():
            acertos += (model(x_adv).argmax(1) == y).sum().item()
        total += x.size(0)
    return acertos / total


@torch.no_grad()
def matriz_confusao(model, loader, device):
    model.eval()
    matriz = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        for real, p in zip(y, pred):
            matriz[real, p] += 1
    return matriz


def relatorio_por_classe(matriz):
    acertos = matriz.diag()
    totais = matriz.sum(1)
    print("\nacuracia por classe (da pior para a melhor):")
    ordem = sorted(range(NUM_CLASSES), key=lambda c: acertos[c] / totais[c])
    for c in ordem:
        # classe que mais "rouba" exemplos desta
        erros = matriz[c].clone()
        erros[c] = 0
        confusa = erros.argmax().item()
        extra = f" (confunde com {CLASSES[confusa]}: {erros[confusa]})" if erros.sum() > 0 else ""
        print(f"  {c:2d} {CLASSES[c]:<48} {acertos[c] / totais[c]:.4f}  [{acertos[c]}/{totais[c]}]{extra}")


def main():
    parser = argparse.ArgumentParser(description="treina a SinalNet no GTSRB")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data", default="data", help="pasta do dataset")
    parser.add_argument("--out", default="sinalnet.pt", help="onde salvar o checkpoint")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--adv-treino", action="store_true",
                         help="treina com adversarial training (PGD), para a rede ficar robusta a ataques")
    parser.add_argument("--adv-eps", type=float, default=0.03)
    parser.add_argument("--adv-alpha", type=float, default=0.01)
    parser.add_argument("--adv-passos", type=int, default=5)
    parser.add_argument("--adv-proporcao", type=float, default=0.7,
                         help="fracao de cada batch que vira adversarial, o resto fica limpo")
    parser.add_argument("--adv-warmup", type=int, default=5,
                         help="epocas de rampa linear do eps do ataque, de 0 ate --adv-eps")
    args = parser.parse_args()
    epocas, saida = args.epochs, args.out
    device = escolher_device()
    print(f"usando {device}")

    adv = None
    if args.adv_treino:
        adv = {
            "eps": args.adv_eps, "alpha": args.adv_alpha, "passos": args.adv_passos,
            "proporcao": args.adv_proporcao, "warmup": args.adv_warmup,
        }
        print(f"adversarial training ativado: eps={adv['eps']} alpha={adv['alpha']} "
              f"passos={adv['passos']} proporcao={adv['proporcao']} warmup={adv['warmup']}")

    loader_treino, loader_teste = criar_loaders(args.data, args.batch_size, args.workers)
    model = SinalNet().to(device)
    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=epocas)

    melhor_acc = 0.0
    for epoca in range(1, epocas + 1):
        inicio = time.time()
        if adv is not None:
            progresso = min(1.0, epoca / max(1, adv["warmup"]))
            adv_epoca = {**adv, "eps": adv["eps"] * progresso}
        else:
            adv_epoca = None
        perda, acc = treinar_epoca(model, loader_treino, criterio, otimizador, device, adv_epoca)
        scheduler.step()
        acc_teste = avaliar(model, loader_teste, device)
        print(f"epoca {epoca}/{epocas} | perda {perda:.4f} | acc treino {acc:.4f} "
              f"| acc teste {acc_teste:.4f} | {time.time() - inicio:.0f}s")

        if acc_teste > melhor_acc:
            melhor_acc = acc_teste
            torch.save({"model": model.state_dict(), "acc": acc_teste, "img_size": IMG_SIZE}, saida)
            print(f"  -> melhor modelo salvo em {saida}")

    print(f"melhor acuracia no teste: {melhor_acc:.4f}")

    # relatorio com o melhor checkpoint, nao com o da ultima epoca
    checkpoint = torch.load(saida, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    relatorio_por_classe(matriz_confusao(model, loader_teste, device))

    acc_pgd = avaliar_robustez(model, loader_teste, device)
    print(f"\nacuracia sob ataque PGD (eps=0.03): {acc_pgd:.4f}")


if __name__ == "__main__":
    # o terminal do Windows nao lida bem com acentos nos nomes das classes
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
