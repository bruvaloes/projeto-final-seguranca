# SinalNet: classificador de placas de trânsito

Trabalho de Segurança Computacional sobre ataques adversariais em IA. Nesta etapa o repositório tem o **modelo de visão computacional**: uma CNN pequena (SinalNet) treinada do zero no [GTSRB](https://benchmark.ini.rub.de/gtsrb_news.html), que reconhece as 43 classes de placas em imagens e vídeos. O ataque fica para a entrega final.

O modelo recebe imagens RGB 48×48 com pixels em [0, 1]. A normalização é feita dentro do próprio modelo, o que deixa ele pronto para receber perturbações direto no espaço de pixels.

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate        # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Treino

```bash
python train.py --epochs 10
```

O dataset é baixado automaticamente para `data/`. O melhor checkpoint (pela acurácia no teste) é salvo em `sinalnet.pt` e, ao final, o script mostra a acurácia por classe e com qual classe cada uma mais se confunde. Outras opções: `--batch-size`, `--lr`, `--data`, `--out`, `--workers`.

## Resultado

| | |
|---|---|
| Acurácia no teste (12.630 imagens) | **97,29%** |
| Treino | 10 épocas, CPU, ~2 min por época |
| Tamanho do modelo | 1,48 M parâmetros, ~6 MB |

O split de treino do `torchvision` tem 26.640 imagens. Trinta classes ficam acima de 97%, e as que mais erram são placas muito parecidas entre si:

- Travessia de pedestres (62%), confundida com Área escolar;
- Velocidade máxima 20 km/h (78%), confundida com 70 km/h;
- Cuidado: gelo ou neve (81%), confundida com Animais silvestres;
- Velocidade máxima 60 km/h (89%), confundida com 80 km/h.

A acurácia de treino (86%) fica abaixo da de teste porque o treino usa augmentation pesado e dropout. Mais épocas devem melhorar o resultado.

## Classificar uma imagem ou um vídeo

```bash
python predict.py placa.jpg            # top-3 com porcentagem
python predict.py video.mp4            # classe ao longo do tempo
python predict.py video.mp4 --a-cada 5 --suavizacao 5
python predict.py --webcam             # teste rápido com a câmera (q para sair)
```

O modelo é um **classificador**, não um detector: ele funciona melhor com a placa ocupando boa parte do quadro.

## Uso a partir da interface

```python
from predict import carregar_modelo, classificar, classificar_frame, classificar_video

model = carregar_modelo("sinalnet.pt")

classificar(model, "placa.jpg")         # aceita caminho ou PIL.Image
# [{"classe": 14, "nome": "Pare", "confianca": 0.99}, ...]

classificar_frame(model, frame)         # frame numpy do OpenCV (BGR)

for r in classificar_video(model, "video.mp4", a_cada_n_frames=5, suavizacao=5):
    print(r["tempo_s"], r["resultado"][0]["nome"])
```

Também estão disponíveis `preparar_imagem` (imagem → tensor `[1, 3, 48, 48]`), `classificar_tensor` (classifica direto um tensor) e `tensor_para_imagem` (tensor → `PIL.Image` ampliada para exibir).

## Arquivos

- `model.py`: nomes das classes, camada de normalização e a SinalNet.
- `train.py`: treino e avaliação.
- `predict.py`: funções de inferência para imagem e vídeo, e a CLI.
- `sinalnet.pt`: pesos treinados.
