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


@torch.no_grad()
def _prob_classe(model, x, classe):
    return F.softmax(model(x), dim=1)[0, classe].item()


@torch.no_grad()
def simba(model, x, classe, direcionado, passos=2000, step=0.2):
    # SimBA (Guo et al., "Simple Black-box Adversarial Attacks"): nunca olha
    # o gradiente do modelo, so consulta a probabilidade de uma classe apos
    # cada tentativa de perturbacao. E o que um atacante sem acesso aos
    # pesos consegue fazer contra um modelo atras de uma API, por exemplo.
    x = x.clone()
    prob_atual = _prob_classe(model, x, classe)
    n = x.numel()
    ordem = torch.randperm(n, device=x.device)
    consultas = 0

    for i in range(min(passos, n)):
        idx = ordem[i].item()
        melhorou = False
        for sinal in (1.0, -1.0):
            tentativa = x.clone()
            tentativa.view(-1)[idx] = (tentativa.view(-1)[idx] + sinal * step).clamp(0, 1)
            prob_tentativa = _prob_classe(model, tentativa, classe)
            consultas += 1

            indo_na_direcao_certa = prob_tentativa > prob_atual if direcionado else prob_tentativa < prob_atual
            if indo_na_direcao_certa:
                x, prob_atual, melhorou = tentativa, prob_tentativa, True
                break
        if not melhorou:
            continue

    return x, consultas


def atacar(model, imagem, classe_alvo=None, top_k=3, passos=2000, step=0.2):
    x = preparar_imagem(imagem).to(DEVICE)
    original = _top_k(probabilidades(model, x), top_k)

    direcionado = classe_alvo is not None
    classe = classe_alvo if direcionado else original[0]["classe"]

    x_adv, consultas = simba(model, x, classe, direcionado, passos, step)
    adversario = _top_k(probabilidades(model, x_adv), top_k)
    return original, adversario, x_adv, consultas


def main():
    parser = argparse.ArgumentParser(
        description="ataque black-box (SimBA, so consulta a saida do modelo) contra a SinalNet")
    parser.add_argument("arquivo", help="imagem da placa")
    parser.add_argument("--modelo", default="sinalnet.pt")
    parser.add_argument("--classe", type=int, default=None, help="classe alvo (ataque direcionado); sem isso, ataca a predicao atual")
    parser.add_argument("--passos", type=int, default=2000, help="tentativas de perturbacao (no maximo, uma por pixel/canal)")
    parser.add_argument("--step", type=float, default=0.2, help="tamanho de cada perturbacao de pixel")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--salvar", default="adversario_blackbox.png")
    args = parser.parse_args()

    model = carregar_modelo(args.modelo)
    original, adversario, x_adv, consultas = atacar(
        model, args.arquivo, args.classe, args.top_k, args.passos, args.step)

    print("predicao original:")
    imprimir_resultado(original)
    print(f"\npredicao apos o ataque black-box ({consultas} consultas ao modelo):")
    imprimir_resultado(adversario)

    tensor_para_imagem(x_adv).save(args.salvar)
    print(f"\nimagem perturbada salva em {args.salvar}")


if __name__ == "__main__":
    main()
