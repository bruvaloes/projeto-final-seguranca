import torch
import torch.nn as nn

IMG_SIZE = 48

# nomes na ordem oficial dos rotulos do GTSRB (0 a 42)
CLASSES = [
    "Velocidade máxima 20 km/h",
    "Velocidade máxima 30 km/h",
    "Velocidade máxima 50 km/h",
    "Velocidade máxima 60 km/h",
    "Velocidade máxima 70 km/h",
    "Velocidade máxima 80 km/h",
    "Fim da velocidade máxima 80 km/h",
    "Velocidade máxima 100 km/h",
    "Velocidade máxima 120 km/h",
    "Proibido ultrapassar",
    "Proibido ultrapassar para caminhões",
    "Preferência no próximo cruzamento",
    "Via preferencial",
    "Dê a preferência",
    "Pare",
    "Proibido trânsito de veículos",
    "Proibido trânsito de caminhões",
    "Sentido proibido",
    "Atenção",
    "Curva perigosa à esquerda",
    "Curva perigosa à direita",
    "Curva dupla",
    "Pista irregular",
    "Pista escorregadia",
    "Estreitamento de pista à direita",
    "Obras",
    "Semáforo",
    "Travessia de pedestres",
    "Área escolar",
    "Travessia de ciclistas",
    "Cuidado: gelo ou neve",
    "Animais silvestres",
    "Fim de todas as restrições",
    "Vire à direita",
    "Vire à esquerda",
    "Siga em frente",
    "Siga em frente ou à direita",
    "Siga em frente ou à esquerda",
    "Mantenha-se à direita",
    "Mantenha-se à esquerda",
    "Rotatória",
    "Fim da proibição de ultrapassar",
    "Fim da proibição de ultrapassar para caminhões",
]
NUM_CLASSES = len(CLASSES)

MEAN = (0.3403, 0.3121, 0.3214)
STD = (0.2724, 0.2608, 0.2669)


# A normalizacao fica dentro do modelo para ele receber pixels em [0, 1].
# Assim o ataque (etapa final) perturba a imagem direto no espaco de pixels
# e so precisa fazer clamp(0, 1), sem desfazer a normalizacao.
class Normalize(nn.Module):
    def __init__(self, mean=MEAN, std=STD):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def bloco_conv(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout(0.2),
    )


class SinalNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.normalize = Normalize()
        self.features = nn.Sequential(
            bloco_conv(3, 32),    # 48 -> 24
            bloco_conv(32, 64),   # 24 -> 12
            bloco_conv(64, 128),  # 12 -> 6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x em [0, 1]; retorna logits
        x = self.normalize(x)
        x = self.features(x)
        return self.classifier(x)
