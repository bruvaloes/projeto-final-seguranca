import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import CLASSES, IMG_SIZE, SinalNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),  # [0, 1], a normalizacao fica no modelo
])


def carregar_modelo(caminho="sinalnet.pt"):
    checkpoint = torch.load(caminho, map_location=DEVICE, weights_only=True)
    model = SinalNet().to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def preparar_imagem(imagem):
    # aceita PIL.Image ou caminho para o arquivo
    if not isinstance(imagem, Image.Image):
        imagem = Image.open(imagem)
    imagem = imagem.convert("RGB")
    return _transform(imagem).unsqueeze(0)


def _top_k(probs, top_k):
    confs, idxs = probs.topk(top_k)
    return [
        {"classe": int(i), "nome": CLASSES[int(i)], "confianca": float(c)}
        for c, i in zip(confs, idxs)
    ]


@torch.no_grad()
def probabilidades(model, x):
    return F.softmax(model(x.to(DEVICE)), dim=1)[0].cpu()


def classificar_tensor(model, x, top_k=3):
    return _top_k(probabilidades(model, x), top_k)


def classificar(model, imagem, top_k=3):
    return classificar_tensor(model, preparar_imagem(imagem), top_k)


def tensor_para_imagem(x, tamanho=256):
    # amplia com NEAREST para os pixels ficarem visiveis na interface
    x = x.detach().cpu().squeeze(0).clamp(0, 1)
    imagem = transforms.functional.to_pil_image(x)
    return imagem.resize((tamanho, tamanho), Image.NEAREST)
