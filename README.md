# SinalNet

Traffic sign classifier built for a Computational Security course project, where we study adversarial attacks against computer vision models. This part of the repo holds the model: a small CNN we call SinalNet, trained from scratch on the [GTSRB](https://benchmark.ini.rub.de/gtsrb_news.html) (German Traffic Sign Recognition Benchmark), a dataset with 43 types of German road signs.

The network takes 48×48 RGB images with pixels between 0 and 1. We keep normalization inside the model itself, not in the preprocessing step, so we can poke at the image's raw pixels directly without having to undo normalization afterward.

The threat model, risk analysis, and the attacks beyond FGSM/PGD (physical patch, black-box) are in [SECURITY.md](SECURITY.md).

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # on Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

This installs plain PyTorch, which is already enough to use the model. The trained weights are in `sinalnet.pt` and work on any machine, with or without a GPU. The code detects the best available option on its own: CUDA (NVIDIA), then MPS (the integrated GPU on Apple Silicon Macs), and CPU as a last resort.

To run the tests (`pip install -r requirements-dev.txt`):

```bash
pytest tests/
```

### Training with an AMD GPU (optional)

We trained on an RX 9060 XT on Windows. For PyTorch to see an AMD card, you need to install the ROCm build from AMD's repository, swapping `gfx1200` for your card's architecture if it's a different one (the list is in the [TheRock repository](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)):

```bash
pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1200]" "torchvision[device-gfx1200]"
pip install opencv-python
```

Even though it's AMD, this PyTorch build shows up as `cuda`, so the code doesn't need any changes. At the start of training, MIOpen prints a few warnings about tuning files it couldn't find, but those can be ignored.

## Training

```bash
python -m sinalnet.train --epochs 10
```

The first time, the dataset is downloaded into the `data/` folder. Every epoch, the model is evaluated on the test set, and only the best checkpoint gets saved to `sinalnet.pt`. At the end, it prints the accuracy per class along with the class it gets confused with most, which helps show where the model fails. You can also pass `--batch-size`, `--lr`, `--data`, `--out`, and `--workers`.

To make the model more robust on real photos and videos, training uses rotation, translation, scale changes, perspective, brightness/contrast variation, a bit of blur, Gaussian noise, and random partial occlusion (`RandomErasing`, simulating dirt, a sticker, or a shadow on the sign) — meant for signs seen under difficult conditions, not just perfect photos. We don't use horizontal flip, because it would turn "turn left" into "turn right".

### Adversarial training (attack-robust training)

```bash
python -m sinalnet.train --epochs 40 --adv-treino --out sinalnet_robusto.pt
```

With `--adv-treino`, each batch is replaced by its adversarial version (generated with PGD, using the model currently being trained) before the optimization step — the idea from Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks". You can tune the strength of the attack used in training with `--adv-eps`, `--adv-alpha`, and `--adv-passos`. At the end, besides the per-class report, it prints the model's accuracy under a PGD attack on the test set, to measure how much more robust it became.

This training is much slower than the normal one (each batch does `--adv-passos` extra forward/backward rounds to generate the attack) and also converges more slowly, needing more epochs. With 80 epochs, an eps warm-up (`--adv-warmup`), and attacking only 70% of each batch (`--adv-proporcao`), we reached **95.50%** clean accuracy and **61.07%** accuracy under a PGD attack (eps=0.03) on the test images — much closer to the normal model's 97.69% than in a first attempt without those adjustments (which plateaued at 77.41%/53.63% with 40 epochs). Details on how well these results generalize (or not) to other attack types are in [SECURITY.md](SECURITY.md).

## Adversarial attacks

```bash
python -m ataques.attack placa.jpg --eps 0.05                          # FGSM, untargeted
python -m ataques.attack placa.jpg --metodo pgd --eps 0.05 --passos 10 # PGD, stronger
python -m ataques.attack placa.jpg --metodo pgd --direcionado --classe 14 --eps 0.08 --passos 20
```

`ataques/attack.py` perturbs the image to fool `SinalNet`, printing the prediction before and after the attack and saving the perturbed version (`--salvar`). Two methods:

- **FGSM** (single step): fast, but a weaker perturbation.
- **PGD** (iterative, `--alpha` and `--passos`): repeats the FGSM step several times, projecting back onto the ball of radius `--eps` at each iteration; considerably stronger.

Without `--classe`, the attack is untargeted: it uses the model's own prediction as the target to move away from. With `--direcionado --classe N`, the attack forces the model to specifically predict class `N`.

## Results

With 10 epochs, the model reached **97.69%** accuracy on the 12,630 test images. On the GPU, each epoch took about 38 seconds (the first two take longer, while MIOpen picks its kernels), and the whole training run took around 9 minutes. We'd trained on CPU before, where each epoch took about 2 minutes and the result was similar, 97.29%. The network has 1.48 million parameters and the weights file is about 6 MB.

Most classes score above 97%. Errors concentrate on signs that look very similar at 48×48, like pedestrian crossing (50%), which the model confuses with right-of-way at the next intersection, and 20 km/h speed limit (70%), which turns into 30 km/h. One detail we found curious is that training accuracy (92.5%) ended up below test accuracy, which happens because the training images go through augmentation and dropout. The `torchvision` training split has 26,640 images.

Since the model has only ever seen German signs, we haven't properly tested it on Brazilian signs yet. The expectation is that it will get the ones that look like the German versions right, such as Stop, Yield, No Entry, and speed limits, and get the warning signs wrong, since those are yellow diamonds in Brazil, as well as the mandatory signs, which are blue in Germany. By default it always answers with one of the 43 classes, even when the sign isn't in the dataset at all — which is why there's a confidence `--limiar` (threshold) in `sinalnet/predict.py` (see the next section).

## Classifying images and videos

```bash
python -m sinalnet.predict placa.jpg            # shows the 3 most likely classes
python -m sinalnet.predict video.mp4            # shows the class throughout the video
python -m sinalnet.predict video.mp4 --a-cada 5 --suavizacao 5
python -m sinalnet.predict --webcam             # opens the camera, q to quit
python -m sinalnet.predict placa.jpg --limiar 0.7
```

The model classifies the whole image and doesn't look for where the sign is, so it works best when the sign takes up a good part of the photo or frame. On video, it analyzes one frame every N and averages the probabilities of the last analyzed frames, so the class doesn't keep flipping from one frame to the next.

Since the model always answers with one of the 43 classes, even when the image isn't a GTSRB sign at all, `--limiar` (default 0.5) marks it as "unknown" whenever the maximum confidence falls below that threshold.

## Using it from an interface

```python
from sinalnet.predict import carregar_modelo, classificar, classificar_frame, classificar_video

model = carregar_modelo("sinalnet.pt")

classificar(model, "placa.jpg")         # accepts a path or a PIL.Image
# [{"classe": 14, "nome": "Pare", "confianca": 0.99}, ...]

classificar_frame(model, frame)         # an OpenCV frame (numpy, BGR)

for r in classificar_video(model, "video.mp4", a_cada_n_frames=5, suavizacao=5):
    print(r["tempo_s"], r["resultado"][0]["nome"])
```

Besides these, `preparar_imagem` turns the image into a `[1, 3, 48, 48]` tensor, `classificar_tensor` classifies directly from a tensor, and `tensor_para_imagem` goes the other way, returning an enlarged `PIL.Image` to show on screen.

## Files

- `sinalnet/`: the project's core (model, training, and inference)
  - `model.py`: class names, the normalization layer, and SinalNet itself
  - `train.py`: training, evaluation, and adversarial training
  - `augmentacao.py`: extra augmentation transforms used in training
  - `predict.py`: inference functions for images and video, plus the command line
- `ataques/`: adversarial attack scripts against the model
  - `attack.py`: FGSM and PGD (requires access to the weights)
  - `patch_attack.py`: physical patch robust to angle/scale/brightness (EoT)
  - `blackbox_attack.py`: query-based attack, no gradient needed (SimBA)
- `analise/`: quantitative evaluation of the attacks and defenses
  - `eval_robustez.py`: accuracy × attack strength curve (FGSM/PGD) on the whole test set
  - `transferencia.py`: tests whether adversarial examples from one model fool the other
  - `avaliar_defesas.py`: tests whether adversarial training (trained only against PGD) also reduces the physical patch and the black-box attack
  - `deteccao.py`: an attempt (unsuccessful, and documented as such) to detect adversarial input
- `tests/`: automated tests (pytest) for `sinalnet/model.py` and `sinalnet/predict.py`
- `sinalnet.pt`: trained weights
- `SECURITY.md`: risk analysis, threat scenario, and MITRE ATLAS mapping
