import numpy as np
import torch
from PIL import Image

from sinalnet.model import NUM_CLASSES, SinalNet
from sinalnet.predict import DEVICE, classificar_tensor, desconhecido, preparar_imagem


def test_preparar_imagem_aceita_pil_e_normaliza_para_0_1():
    imagem = Image.fromarray((np.random.rand(48, 48, 3) * 255).astype("uint8"))
    x = preparar_imagem(imagem)
    assert x.shape == (1, 3, 48, 48)
    assert x.min() >= 0 and x.max() <= 1


def test_classificar_tensor_devolve_top_k_ordenado():
    model = SinalNet().to(DEVICE)
    x = torch.rand(1, 3, 48, 48)
    resultado = classificar_tensor(model, x, top_k=5)
    assert len(resultado) == 5
    confiancas = [r["confianca"] for r in resultado]
    assert confiancas == sorted(confiancas, reverse=True)
    assert all(0 <= r["classe"] < NUM_CLASSES for r in resultado)


def test_desconhecido_abaixo_do_limiar():
    resultado = [{"classe": 0, "nome": "x", "confianca": 0.3}]
    assert desconhecido(resultado, limiar=0.5)


def test_nao_desconhecido_acima_do_limiar():
    resultado = [{"classe": 0, "nome": "x", "confianca": 0.8}]
    assert not desconhecido(resultado, limiar=0.5)
