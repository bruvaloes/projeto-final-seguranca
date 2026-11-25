import argparse
import os
import sys
from collections import deque

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import CLASSES, IMG_SIZE, SinalNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXTENSOES_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# o modelo sempre responde uma das 43 classes, mesmo para uma imagem que nao
# e nenhuma placa do GTSRB; abaixo deste limiar tratamos como "desconhecido"
LIMIAR_PADRAO = 0.5

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


def desconhecido(resultado, limiar=LIMIAR_PADRAO):
    return resultado[0]["confianca"] < limiar


def tensor_para_imagem(x, tamanho=256):
    # amplia com NEAREST para os pixels ficarem visiveis na interface
    x = x.detach().cpu().squeeze(0).clamp(0, 1)
    imagem = transforms.functional.to_pil_image(x)
    return imagem.resize((tamanho, tamanho), Image.NEAREST)


# O modelo e um classificador, nao um detector: funciona quando a placa
# ocupa boa parte do quadro (camera apontada para a placa).
def frame_para_imagem(frame, bgr=True):
    if bgr:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def classificar_frame(model, frame, bgr=True, top_k=3):
    return classificar(model, frame_para_imagem(frame, bgr), top_k)


def classificar_video(model, caminho, a_cada_n_frames=5, suavizacao=5, top_k=3):
    cap = cv2.VideoCapture(caminho)
    if not cap.isOpened():
        raise ValueError(f"nao foi possivel abrir o video {caminho}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # media das probabilidades dos ultimos K frames analisados, para a
    # predicao nao ficar "piscando" entre classes de um frame para o outro
    ultimas = deque(maxlen=max(1, suavizacao))

    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if n % a_cada_n_frames == 0:
                x = preparar_imagem(frame_para_imagem(frame))
                ultimas.append(probabilidades(model, x))
                media = torch.stack(list(ultimas)).mean(0)
                yield {
                    "frame": n,
                    "tempo_s": n / fps,
                    "resultado": _top_k(media, top_k),
                }
            n += 1
    finally:
        cap.release()


def rodar_webcam(model, camera=0, suavizacao=5, limiar=LIMIAR_PADRAO):
    # so para testes locais; a interface oficial e a da Bruna
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise ValueError(f"nao foi possivel abrir a camera {camera}")
    ultimas = deque(maxlen=max(1, suavizacao))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ultimas.append(probabilidades(model, preparar_imagem(frame_para_imagem(frame))))
            resultado = _top_k(torch.stack(list(ultimas)).mean(0), 1)
            top = resultado[0]
            if desconhecido(resultado, limiar):
                texto = f"desconhecido ({top['confianca'] * 100:.0f}%)"
            else:
                # cv2.putText nao desenha acentos, entao mostramos o numero da classe junto
                texto = f"{top['classe']}: {top['nome']} ({top['confianca'] * 100:.0f}%)"
            cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("SinalNet - q para sair", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def imprimir_resultado(resultado, limiar=LIMIAR_PADRAO):
    if desconhecido(resultado, limiar):
        print(f"  desconhecido (confianca maxima {resultado[0]['confianca'] * 100:.2f}% abaixo do limiar)")
    for r in resultado:
        print(f"  {r['confianca'] * 100:6.2f}%  {r['nome']} (classe {r['classe']})")


def main():
    parser = argparse.ArgumentParser(description="classifica placas de transito com a SinalNet")
    parser.add_argument("arquivo", nargs="?", help="imagem ou video da placa")
    parser.add_argument("--modelo", default="sinalnet.pt")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--a-cada", type=int, default=5, help="no video, analisa um frame a cada N")
    parser.add_argument("--suavizacao", type=int, default=5, help="no video, media dos ultimos K frames (1 desliga)")
    parser.add_argument("--webcam", action="store_true", help="classifica ao vivo pela camera")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--limiar", type=float, default=LIMIAR_PADRAO,
                         help="confianca minima para nao tratar como desconhecido")
    args = parser.parse_args()
    if not args.webcam and not args.arquivo:
        parser.error("informe um arquivo ou use --webcam")

    model = carregar_modelo(args.modelo)

    if args.webcam:
        rodar_webcam(model, args.camera, args.suavizacao, args.limiar)
    elif os.path.splitext(args.arquivo)[1].lower() in EXTENSOES_VIDEO:
        for r in classificar_video(model, args.arquivo, args.a_cada, args.suavizacao, top_k=1):
            top = r["resultado"][0]
            nome = "desconhecido" if desconhecido(r["resultado"], args.limiar) else top["nome"]
            print(f"{r['tempo_s']:6.2f}s  frame {r['frame']:5d}  {top['confianca'] * 100:6.2f}%  {nome}")
    else:
        print(args.arquivo)
        imprimir_resultado(classificar(model, args.arquivo, args.top_k), args.limiar)


if __name__ == "__main__":
    # o terminal do Windows nao lida bem com acentos nos nomes das classes
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
