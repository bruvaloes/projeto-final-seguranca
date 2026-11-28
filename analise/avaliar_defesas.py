import argparse
import random

from ataques.blackbox_attack import simba
from ataques.patch_attack import treinar_patch
from sinalnet.model import escolher_device
from sinalnet.predict import carregar_modelo, tensor_para_imagem
from sinalnet.train import criar_loaders


def avaliar_patch(model, dataset, device, indices, **kwargs):
    sucesso = 0
    for i in indices:
        x, y = dataset[i]
        imagem = tensor_para_imagem(x.unsqueeze(0), tamanho=48)
        _, _, _, x_final = treinar_patch(model, imagem, classe_alvo=None, **kwargs)
        pred = model(x_final).argmax(1).item()
        if pred != y:
            sucesso += 1
    return sucesso / len(indices)


def avaliar_blackbox(model, dataset, device, indices, **kwargs):
    sucesso = 0
    for i in indices:
        x, y = dataset[i]
        x = x.unsqueeze(0).to(device)
        pred_original = model(x).argmax(1).item()
        x_adv, _ = simba(model, x, pred_original, direcionado=False, **kwargs)
        pred = model(x_adv).argmax(1).item()
        if pred != y:
            sucesso += 1
    return sucesso / len(indices)


def main():
    parser = argparse.ArgumentParser(
        description="testa se o adversarial training (treinado so contra PGD) tambem reduz o patch fisico e o black-box")
    parser.add_argument("--normal", default="sinalnet.pt")
    parser.add_argument("--robusto", default="sinalnet_robusto.pt")
    parser.add_argument("--amostra", type=int, default=20)
    parser.add_argument("--data", default="data")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = escolher_device()
    print(f"usando {device}")
    _, loader_teste = criar_loaders(args.data)
    dataset = loader_teste.dataset

    random.seed(args.seed)
    indices = random.sample(range(len(dataset)), args.amostra)

    for nome, caminho in [("normal", args.normal), ("robusto", args.robusto)]:
        model = carregar_modelo(caminho)
        taxa_patch = avaliar_patch(model, dataset, device, indices)
        taxa_bb = avaliar_blackbox(model, dataset, device, indices)
        print(f"\n{nome} ({caminho}):")
        print(f"  taxa de sucesso do patch fisico:   {taxa_patch:.2%}")
        print(f"  taxa de sucesso do black-box:      {taxa_bb:.2%}")


if __name__ == "__main__":
    main()
