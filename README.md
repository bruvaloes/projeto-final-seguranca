# SinalNet

Classificador de placas de trânsito feito para o trabalho de Segurança Computacional, em que estudamos ataques adversariais contra modelos de visão computacional. Aqui fica a parte do modelo: uma CNN pequena, que chamamos de SinalNet, treinada do zero no [GTSRB](https://benchmark.ini.rub.de/gtsrb_news.html) (German Traffic Sign Recognition Benchmark), um dataset com 43 tipos de placas alemãs.

A rede recebe imagens RGB de 48×48 com os pixels entre 0 e 1. Deixamos a normalização dentro do próprio modelo, e não no pré-processamento, porque assim dá para mexer direto nos pixels da imagem sem ter que desfazer a normalização depois.

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate        # no Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Isso instala o PyTorch padrão, que já é suficiente para usar o modelo. Os pesos treinados estão em `sinalnet.pt` e funcionam em qualquer máquina, com ou sem GPU. O código detecta sozinho a melhor opção disponível: CUDA (NVIDIA), depois MPS (GPU integrada dos Macs com Apple Silicon) e por último CPU.

Para rodar os testes (`pip install -r requirements-dev.txt`):

```bash
pytest tests/
```

### Treinando com GPU AMD (opcional)

Treinamos numa RX 9060 XT no Windows. Para o PyTorch enxergar uma placa AMD é preciso instalar a versão com ROCm a partir do repositório da AMD, trocando `gfx1200` pela arquitetura da sua placa se for outra (a lista está no [repositório TheRock](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)):

```bash
pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1200]" "torchvision[device-gfx1200]"
pip install opencv-python
```

Mesmo sendo AMD, essa versão do PyTorch aparece como `cuda`, então o código não precisa de nenhuma mudança. No começo do treino o MIOpen mostra alguns avisos sobre arquivos de tuning que não encontrou, mas eles podem ser ignorados.

## Treino

```bash
python train.py --epochs 10
```

Na primeira vez o dataset é baixado para a pasta `data/`. A cada época o modelo é avaliado no conjunto de teste e só o melhor checkpoint fica salvo em `sinalnet.pt`. No final aparece a acurácia de cada classe junto com a classe com que ela mais se confunde, o que ajuda a ver onde o modelo erra. Também dá para passar `--batch-size`, `--lr`, `--data`, `--out` e `--workers`.

Para deixar o modelo mais robusto em fotos e vídeos reais, o treino usa rotação, translação, mudança de escala, perspectiva, variação de brilho e contraste e um pouco de desfoque. Não usamos flip horizontal porque ele transformaria "vire à esquerda" em "vire à direita".

## Resultados

Com 10 épocas o modelo chegou a **97,69%** de acurácia nas 12.630 imagens de teste. Na GPU cada época levou uns 38 segundos (as duas primeiras demoram mais, enquanto o MIOpen escolhe os kernels), e o treino inteiro ficou em torno de 9 minutos. Antes tínhamos treinado na CPU, onde cada época levava uns 2 minutos e o resultado foi parecido, 97,29%. A rede tem 1,48 milhão de parâmetros e o arquivo de pesos ocupa uns 6 MB.

A maioria das classes passa de 97%. Os erros se concentram em placas que ficam muito parecidas em 48×48, como travessia de pedestres (50%), que o modelo confunde com preferência no próximo cruzamento, e velocidade máxima de 20 km/h (70%), que vira 30 km/h. Um detalhe que achamos curioso é que a acurácia no treino (92,5%) ficou abaixo da de teste, o que acontece porque as imagens de treino passam pelo augmentation e pelo dropout. O split de treino do `torchvision` tem 26.640 imagens.

Como o modelo só viu placas alemãs, ainda não testamos direito com placas brasileiras. A expectativa é que ele acerte as que são parecidas com as de lá, como Pare, Dê a preferência, Sentido proibido e velocidade máxima, e erre as de advertência, que no Brasil são losangos amarelos, e as de obrigação, que na Alemanha são azuis. Por padrão ele sempre responde uma das 43 classes, mesmo quando a placa não existe no dataset — por isso o `--limiar` de confiança em `predict.py` (veja a seção seguinte).

## Classificando imagens e vídeos

```bash
python predict.py placa.jpg            # mostra as 3 classes mais prováveis
python predict.py video.mp4            # mostra a classe ao longo do vídeo
python predict.py video.mp4 --a-cada 5 --suavizacao 5
python predict.py --webcam             # abre a câmera, q para sair
python predict.py placa.jpg --limiar 0.7
```

O modelo classifica a imagem inteira e não procura onde está a placa, por isso funciona melhor quando a placa ocupa boa parte da foto ou do quadro. No vídeo ele analisa um frame a cada N e tira a média das probabilidades dos últimos frames analisados, para a classe não ficar trocando de um frame para o outro.

Como o modelo sempre responde uma das 43 classes, mesmo quando a imagem não é nenhuma placa do GTSRB, `--limiar` (padrão 0,5) marca como "desconhecido" sempre que a confiança máxima ficar abaixo dele.

## Usando na interface

```python
from predict import carregar_modelo, classificar, classificar_frame, classificar_video

model = carregar_modelo("sinalnet.pt")

classificar(model, "placa.jpg")         # aceita caminho ou PIL.Image
# [{"classe": 14, "nome": "Pare", "confianca": 0.99}, ...]

classificar_frame(model, frame)         # frame do OpenCV (numpy, BGR)

for r in classificar_video(model, "video.mp4", a_cada_n_frames=5, suavizacao=5):
    print(r["tempo_s"], r["resultado"][0]["nome"])
```

Além dessas, `preparar_imagem` transforma a imagem num tensor `[1, 3, 48, 48]`, `classificar_tensor` classifica direto a partir de um tensor e `tensor_para_imagem` faz o caminho de volta, devolvendo uma `PIL.Image` ampliada para mostrar na tela.

## Arquivos

- `model.py`: nomes das classes, camada de normalização e a SinalNet
- `train.py`: treino e avaliação
- `predict.py`: funções de inferência para imagem e vídeo e a linha de comando
- `sinalnet.pt`: pesos treinados
- `tests/`: testes automatizados (pytest) de `model.py` e `predict.py`
