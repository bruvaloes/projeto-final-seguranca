import argparse
import random

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from predict import (
    DEVICE,
    _top_k,
    carregar_modelo,
    imprimir_resultado,
    preparar_imagem,
    probabilidades,
    tensor_para_imagem,
)


def _compor(x, patch, tamanho, pos, angulo, escala, brilho):
    # cola o patch num canvas do tamanho da imagem, aplica uma transformacao
    # aleatoria (rotacao/translacao/escala) e so entao mistura com a imagem,
    # simulando a variacao de angulo e distancia de uma foto real da placa
    canvas = torch.zeros_like(x)
    mascara = torch.zeros_like(x)
    canvas[:, :, :tamanho, :tamanho] = (patch * brilho).clamp(0, 1)
    mascara[:, :, :tamanho, :tamanho] = 1.0

    canvas = TF.affine(canvas, angle=angulo, translate=pos, scale=escala, shear=0)
    mascara = TF.affine(mascara, angle=angulo, translate=pos, scale=escala, shear=0)

    return x * (1 - mascara) + canvas * mascara


def treinar_patch(model, imagem, tamanho=20, classe_alvo=None, passos=300, lr=0.08,
                   rotacao_max=5, translacao_max=3, escala_min=0.95, escala_max=1.05,
                   amostras_por_passo=8):
    x = preparar_imagem(imagem).to(DEVICE)
    original = _top_k(probabilidades(model, x), 3)

    direcionado = classe_alvo is not None
    rotulo = classe_alvo if direcionado else original[0]["classe"]
    y = torch.tensor([rotulo], device=DEVICE)
    sinal = -1 if direcionado else 1

    patch = torch.rand(1, 3, tamanho, tamanho, device=DEVICE, requires_grad=True)

    for _ in range(passos):
        # expectation over transformation: tira a media do gradiente sobre
        # varias amostras de transformacao aleatoria, nao so uma, senao o
        # ruido de uma amostra por passo domina e o patch nao converge
        grad_acumulado = torch.zeros_like(patch)
        for _ in range(amostras_por_passo):
            pos = (random.uniform(-translacao_max, translacao_max),
                   random.uniform(-translacao_max, translacao_max))
            angulo = random.uniform(-rotacao_max, rotacao_max)
            escala = random.uniform(escala_min, escala_max)
            brilho = random.uniform(0.7, 1.3)

            x_adv = _compor(x, patch, tamanho, pos, angulo, escala, brilho)
            logits = model(x_adv)
            perda = F.cross_entropy(logits, y)
            grad_acumulado += torch.autograd.grad(perda, patch)[0]

        with torch.no_grad():
            patch += sinal * lr * (grad_acumulado / amostras_por_passo).sign()
            patch.clamp_(0, 1)

    # avalia o patch "fixo" no centro, sem a transformacao aleatoria do treino
    x_final = _compor(x, patch.detach(), tamanho, (0, 0), 0, 1.0, 1.0)
    adversario = _top_k(probabilidades(model, x_final), 3)
    return original, adversario, patch.detach(), x_final


def main():
    parser = argparse.ArgumentParser(
        description="ataque de patch adversarial (fisico, robusto a angulo/escala/brilho) contra a SinalNet")
    parser.add_argument("arquivo", help="imagem da placa")
    parser.add_argument("--modelo", default="sinalnet.pt")
    parser.add_argument("--tamanho", type=int, default=20, help="lado do patch em pixels (de 48)")
    parser.add_argument("--classe", type=int, default=None, help="classe alvo (ataque direcionado); sem isso, ataca a predicao atual")
    parser.add_argument("--passos", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--amostras-por-passo", type=int, default=8,
                         help="transformacoes aleatorias (EoT) usadas para calcular o gradiente a cada passo")
    parser.add_argument("--salvar", default="placa_com_patch.png")
    parser.add_argument("--salvar-patch", default="patch.png")
    args = parser.parse_args()

    model = carregar_modelo(args.modelo)
    original, adversario, patch, x_final = treinar_patch(
        model, args.arquivo, args.tamanho, args.classe, args.passos, args.lr,
        amostras_por_passo=args.amostras_por_passo)

    print("predicao original:")
    imprimir_resultado(original)
    print(f"\npredicao com o patch ({args.tamanho}x{args.tamanho}px):")
    imprimir_resultado(adversario)

    tensor_para_imagem(x_final).save(args.salvar)
    tensor_para_imagem(patch, tamanho=128).save(args.salvar_patch)
    print(f"\nimagem com patch salva em {args.salvar}, patch isolado em {args.salvar_patch}")


if __name__ == "__main__":
    main()
