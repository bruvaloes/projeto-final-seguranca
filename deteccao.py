import argparse

import torch

from predict import DEVICE, _top_k, carregar_modelo, imprimir_resultado, preparar_imagem, probabilidades


@torch.no_grad()
def detectar_adversarial(model, x, amostras=15, sigma=0.03, limiar_discordancia=0.3):
    # hipotese: uma imagem adversarial fica numa fronteira de decisao fina
    # (foi perturbada justamente pra cruzar essa fronteira com a menor
    # distancia possivel). um pouco de ruido aleatorio por cima tende a
    # jogar a predicao de volta pra fronteira, entao a predicao fica
    # instavel. uma imagem limpa, bem distante da fronteira, deveria
    # manter a mesma predicao mesmo com ruido pequeno.
    model.eval()
    pred_original = model(x).argmax(1).item()

    discordancias = 0
    for _ in range(amostras):
        ruido = torch.randn_like(x) * sigma
        x_ruido = (x + ruido).clamp(0, 1)
        pred_ruido = model(x_ruido).argmax(1).item()
        if pred_ruido != pred_original:
            discordancias += 1

    taxa = discordancias / amostras
    return taxa > limiar_discordancia, taxa


def main():
    parser = argparse.ArgumentParser(
        description="detector simples de entrada adversarial: checa se a predicao e instavel sob ruido")
    parser.add_argument("arquivo")
    parser.add_argument("--modelo", default="sinalnet.pt")
    parser.add_argument("--amostras", type=int, default=15)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--limiar", type=float, default=0.3)
    args = parser.parse_args()

    model = carregar_modelo(args.modelo)
    x = preparar_imagem(args.arquivo).to(DEVICE)

    suspeita, taxa = detectar_adversarial(model, x, args.amostras, args.sigma, args.limiar)
    print("predicao:")
    imprimir_resultado(_top_k(probabilidades(model, x), 3))
    print(f"\ntaxa de discordancia sob ruido: {taxa:.2f} ({'suspeita de ataque' if suspeita else 'parece normal'})")


if __name__ == "__main__":
    main()
