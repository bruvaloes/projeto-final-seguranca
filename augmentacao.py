import random

import torch


def ruido_gaussiano(x, intensidade_max=0.08):
    # ruido de sensor/compressao, mais forte em fotos de baixa qualidade
    return (x + torch.randn_like(x) * random.uniform(0, intensidade_max)).clamp(0, 1)
