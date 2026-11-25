import argparse

import torch
import torch.nn.functional as F

from predict import (
    DEVICE,
    _top_k,
    carregar_modelo,
    imprimir_resultado,
    preparar_imagem,
    probabilidades,
    tensor_para_imagem,
)


def fgsm(model, x, y, eps):
    x = x.clone().detach().to(DEVICE).requires_grad_(True)
    logits = model(x)
    perda = F.cross_entropy(logits, y)
    perda.backward()
    return (x + eps * x.grad.sign()).clamp(0, 1).detach()


def atacar(model, imagem, eps=0.03, classe_real=None, top_k=3):
    x = preparar_imagem(imagem)
    original = _top_k(probabilidades(model, x), top_k)

    # sem rotulo verdadeiro, ataca a propria predicao do modelo (nao-direcionado)
    rotulo = classe_real if classe_real is not None else original[0]["classe"]
    y = torch.tensor([rotulo], device=DEVICE)

    x_adv = fgsm(model, x, y, eps)
    adversario = _top_k(probabilidades(model, x_adv), top_k)
    return original, adversario, x_adv


def main():
    parser = argparse.ArgumentParser(description="ataque FGSM contra a SinalNet")
    parser.add_argument("arquivo", help="imagem da placa")
    parser.add_argument("--modelo", default="sinalnet.pt")
    parser.add_argument("--eps", type=float, default=0.03, help="tamanho da perturbacao (0 a 1)")
    parser.add_argument("--classe", type=int, default=None,
                         help="classe verdadeira (opcional); sem isso ataca a predicao atual do modelo")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--salvar", default="adversario.png", help="onde salvar a imagem perturbada")
    args = parser.parse_args()

    model = carregar_modelo(args.modelo)
    original, adversario, x_adv = atacar(model, args.arquivo, args.eps, args.classe, args.top_k)

    print("predicao original:")
    imprimir_resultado(original)
    print(f"\npredicao apos FGSM (eps={args.eps}):")
    imprimir_resultado(adversario)

    tensor_para_imagem(x_adv).save(args.salvar)
    print(f"\nimagem perturbada salva em {args.salvar}")


if __name__ == "__main__":
    main()
