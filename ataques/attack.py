import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predict import (
    DEVICE,
    _top_k,
    carregar_modelo,
    imprimir_resultado,
    preparar_imagem,
    probabilidades,
    tensor_para_imagem,
)


def fgsm(model, x, y, eps, direcionado=False):
    # x e y ja devem estar no mesmo device do modelo
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    perda = F.cross_entropy(logits, y)
    perda.backward()
    # nao-direcionado: sobe a perda (afasta de y); direcionado: desce (aproxima de y)
    sinal = -1 if direcionado else 1
    return (x + sinal * eps * x.grad.sign()).clamp(0, 1).detach()


def pgd(model, x, y, eps, alpha, passos, direcionado=False):
    x_orig = x.clone().detach()
    x_adv = x_orig.clone()
    sinal = -1 if direcionado else 1
    for _ in range(passos):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        perda = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(perda, x_adv)[0]
        x_adv = (x_adv + sinal * alpha * grad.sign()).detach()
        # projeta de volta na bola eps em torno da imagem original
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps).clamp(0, 1)
    return x_adv


def atacar(model, imagem, eps=0.03, classe_real=None, top_k=3,
           metodo="fgsm", alpha=None, passos=10, direcionado=False):
    x = preparar_imagem(imagem).to(DEVICE)
    original = _top_k(probabilidades(model, x), top_k)

    if direcionado:
        if classe_real is None:
            raise ValueError("ataque direcionado precisa de --classe (a classe que se quer forcar)")
        rotulo = classe_real
    else:
        # sem rotulo verdadeiro, ataca a propria predicao do modelo
        rotulo = classe_real if classe_real is not None else original[0]["classe"]
    y = torch.tensor([rotulo], device=DEVICE)

    if metodo == "fgsm":
        x_adv = fgsm(model, x, y, eps, direcionado)
    else:
        x_adv = pgd(model, x, y, eps, alpha or eps / 4, passos, direcionado)

    adversario = _top_k(probabilidades(model, x_adv), top_k)
    return original, adversario, x_adv


def main():
    parser = argparse.ArgumentParser(description="ataques adversariais (FGSM/PGD) contra a SinalNet")
    parser.add_argument("arquivo", help="imagem da placa")
    parser.add_argument("--modelo", default="sinalnet.pt")
    parser.add_argument("--metodo", choices=["fgsm", "pgd"], default="fgsm")
    parser.add_argument("--eps", type=float, default=0.03, help="tamanho maximo da perturbacao (0 a 1)")
    parser.add_argument("--alpha", type=float, default=None, help="passo do PGD (padrao eps/4)")
    parser.add_argument("--passos", type=int, default=10, help="iteracoes do PGD")
    parser.add_argument("--classe", type=int, default=None,
                         help="sem --direcionado: classe verdadeira (opcional). "
                              "com --direcionado: classe que se quer forcar")
    parser.add_argument("--direcionado", action="store_true",
                         help="forca o modelo a prever --classe, em vez de so errar")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--salvar", default="adversario.png", help="onde salvar a imagem perturbada")
    args = parser.parse_args()

    model = carregar_modelo(args.modelo)
    original, adversario, x_adv = atacar(
        model, args.arquivo, args.eps, args.classe, args.top_k,
        args.metodo, args.alpha, args.passos, args.direcionado,
    )

    print("predicao original:")
    imprimir_resultado(original)
    print(f"\npredicao apos {args.metodo.upper()} (eps={args.eps}):")
    imprimir_resultado(adversario)

    tensor_para_imagem(x_adv).save(args.salvar)
    print(f"\nimagem perturbada salva em {args.salvar}")


if __name__ == "__main__":
    main()
