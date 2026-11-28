# Security analysis

This document treats SinalNet as the target of a risk analysis, not as a machine learning project: the goal is to understand what an attacker can do against the model, how feasible that is, and what mitigates each risk. The code (`ataques/attack.py`, `sinalnet/train.py`, `sinalnet/predict.py`) is the tool used to investigate these questions, not the final product of the work.

## Threat model and risk

| | |
|---|---|
| **Asset** | SinalNet's classification decision — which sign the system "thinks" it's looking at. |
| **Threat** | An attacker who can alter the input image (digitally, or by perturbing the physical sign) to force a classification different from the correct one. |
| **Vulnerability** | The model is a neural network trained by gradient descent: small perturbations aligned with the loss gradient are enough to change the prediction, even when the image is still, to a human, clearly the same sign (`ataques/attack.py`, FGSM and PGD). |
| **Feasibility** | High on the undefended model: with access to the weights (white-box), PGD already changes the prediction with small perturbations (eps ~0.03-0.08 on pixels normalized between 0 and 1) in a few iterations. We discuss feasibility under more restricted access (no gradient, no physical access) in the following sections of this document. |
| **Impact** | Depends on where the decision is consumed — see [Threat scenario](#threat-scenario) below. In general, any system that decides a safety-relevant action (stop, don't overtake, respect the speed limit) based on this classification propagates the error downstream. |
| **Mitigation** | *Adversarial training* (`train.py --adv-treino`) reduces the attack's feasibility, at the cost of clean accuracy — see the "Adversarial training" section of the [README](README.md). This is risk reduction, not elimination: the README's Results section shows the robust model still gets it wrong under attack, just less often. |

### Why this is different from an ordinary bug

A traditional software bug (buffer overflow, SQL injection) has a fix that eliminates it. Adversarial robustness doesn't work that way: there's no "patch" that makes the network immune to any perturbation within the attacker's budget — the best you can do is shift the decision boundary so that perturbations of the same size require more computational effort from the attacker. That's why the right metric isn't "does the attack work? yes/no", but rather "how much does it cost (in eps, in queries, in access) to mount the attack, and what does that cost the robust model in return".

## Threat scenario

To discuss impact concretely, we imagine SinalNet as the sign-recognition stage of an ADAS (advanced driver-assistance system) or self-driving car pipeline: camera → preprocessing → **sign classifier** → decision module (brake, respect the speed limit, don't overtake). A wrong classification doesn't stay isolated: it propagates into a vehicle action.

Two quite different attack surfaces, with attackers of quite different capability:

### 1. Digital surface — someone in the image's path

**Attacker**: has access to the data pipeline between the camera and the classifier (e.g., a compromised software component, or an attack on the camera/driver itself). Can alter the pixels before they reach the model.

**Capability**: if they also have access to the model's weights (white-box — the strongest case, and the one `ataques/attack.py` assumes with FGSM/PGD), the needed perturbation is small (eps ~0.03-0.08) and quick to compute. This is the easiest scenario to simulate, and the one we've already implemented.

**Realism**: high *if* the attacker has already compromised something in the pipeline — but at that point they already have substantial access to the system, so the sign classification may not be the easiest link to attack. This attack is more relevant as a study of the model's own robustness than as a standalone intrusion vector.

### 2. Physical surface — someone with only access to the sign

**Attacker**: has no access to any of the car's software, and can only alter the physical world — sticking a sticker on the sign, for example. Doesn't need to compromise anything digital.

**Capability**: more restricted, and more interesting from a security standpoint, because it's a real and cheap attack vector (anyone can get close to a sign). But it's technically harder: the perturbation needs to survive variations in angle, distance, lighting, and the camera's capture — a pixel perturbation calculated for a single photo (like standard FGSM/PGD) usually doesn't survive those transformations. This is addressed in the "Physical attack" section below.

**Realism**: this is the scenario cited in the ML security literature (e.g., stickers on Stop signs) as the most concerning in practice, precisely because it requires no digital access at all — just physical proximity to the sign.

### Which one matters more for this project

The physical surface is the one that best justifies calling this a "security project": it doesn't depend on a strong assumption (access to the model's weights) to be a real risk. The digital surface is easier to demonstrate and quantify, and serves as the technical foundation (it's what the already-implemented FGSM/PGD attacks exploit), but its practical impact depends on an attacker who already has another entry point into the system.

## Physical attack (adversarial patch)

`ataques/patch_attack.py` implements the physical surface described above: instead of perturbing the whole image (FGSM/PGD), it optimizes a small square sticker — a patch — stuck on a fixed corner of the sign, which needs to keep fooling the model even under variations in angle, distance, and lighting of the photo.

```bash
python -m ataques.patch_attack placa.jpg                                   # untargeted
python -m ataques.patch_attack placa.jpg --classe 14 --tamanho 24 --passos 600  # forces "Stop"
```

The technique is **Expectation over Transformation** (EoT, from Athalye et al., "Synthesizing Robust Adversarial Examples"): at each step of training the patch, instead of computing the gradient on a single version of the image, we average the gradient over `--amostras-por-passo` versions with random rotation, translation, scale, and brightness (`_compor` in `ataques/patch_attack.py`). This is what sets this attack apart from FGSM/PGD: the patch isn't optimized for one specific photo, it's optimized to survive a distribution of capture conditions — the real requirement for a physical sticker.

**Finding relevant to this project**: the untargeted attack (just forcing any error) converges fast and with high confidence even with a small EoT range. The **targeted** attack (forcing a specific class, e.g. making the model see "Stop" on any other sign) needs a bigger patch and more steps to converge reliably — with `--tamanho 16` and a wider EoT it doesn't converge well; with `--tamanho 24` (a quarter of the image's area) and 600 steps, it reaches 98.6% confidence in the forced class. In other words: **it's considerably easier for an attacker to cause some error than to choose exactly which error to cause** — a risk asymmetry worth noting in the analysis: even an attacker without fine control can still deny correct recognition of the sign.

## Black-box attack (query-based)

All the previous attacks (FGSM, PGD, the patch) assume access to the model's weights to compute the gradient — **white-box**. A real attacker against a deployed system (behind an API, or a binary they couldn't extract the weights from) usually doesn't have that. `ataques/blackbox_attack.py` implements **SimBA** (Guo et al., "Simple Black-box Adversarial Attacks"): at each step, it tries perturbing a random pixel up or down and only checks whether the probability of the class of interest improved — it never uses `.backward()` or accesses the gradient, it only queries the model's output (`model(x)`), which is exactly what an attacker with only API access could do.

```bash
python -m ataques.blackbox_attack placa.jpg                                    # untargeted
python -m ataques.blackbox_attack placa.jpg --classe 14 --passos 6912 --step 0.3  # forces "Stop"
```

**Finding relevant to this project**: the attack works (tested: untargeted reaches 95%+ confidence in a wrong class, targeted reaches 99.96% forcing "Stop"), but it costs far more **model queries** than gradient calls would have cost in a white-box attack — in testing, the untargeted version used ~4,700 queries, and the targeted one needed to scan the whole image, ~11,400 queries, to converge well. This is the central point of a black-box attack: it's feasible, but it has a **measurable and observable cost** (number of calls to the model) that doesn't exist in the white-box digital attack or the physical attack. That number of queries is exactly what the "restrict the number of model queries" mitigation (see [MITRE ATLAS mapping](#mitre-atlas-mapping)) tries to limit — a system that detects and blocks a client making thousands of queries on the same image rules out this specific attack, without even touching the model.

## MITRE ATLAS mapping

[MITRE ATLAS](https://atlas.mitre.org/) is a knowledge base of tactics and techniques for attacks against AI/ML systems, in the same spirit as ATT&CK for traditional security. Mapping this project's attacks onto that taxonomy helps frame the work as security (language and categories recognized by the community) rather than as an isolated ML experiment.

> **Academic honesty note**: the IDs below come from searches and third-party mirrors of ATLAS, not from a direct, complete consultation of the official site (`atlas.mitre.org`) while writing this document. The tactic/technique names and the mapping's logic are correct; before citing a specific ID (e.g., in a presentation), it's worth confirming it on the [official matrix](https://atlas.mitre.org/matrices/ATLAS).

| Our attack/code | ATLAS tactic | ATLAS technique |
|---|---|---|
| `ataques/attack.py` (FGSM/PGD) — attack with access to the weights | AML.TA0001 *ML Attack Staging* | AML.T0043 *Craft Adversarial Data* → sub-technique AML.T0043.000 *White-Box Optimization* |
| `ataques/patch_attack.py` (physical patch, EoT) | AML.TA0001 *ML Attack Staging* | AML.T0043 *Craft Adversarial Data* → AML.T0043.000 *White-Box Optimization* (uses gradient; the attack's "delivery" is what's physical, not digital) |
| `ataques/blackbox_attack.py` (SimBA) | AML.TA0001 *ML Attack Staging* | AML.T0043 *Craft Adversarial Data* → sub-technique AML.T0043.001 *Black-Box Optimization* |
| Any of the above attacks, from the point of view of the end effect (the classification coming out wrong) | Impact/evasion tactic (varies by mapping) | AML.T0015 *Evade ML Model* |

| Our mitigation/code | ATLAS mitigation |
|---|---|
| `train.py --adv-treino` (adversarial training) | AML.M0003 *Model Hardening* |
| `--limiar` in `sinalnet/predict.py` ("unknown" class) | A weak approximation of AML.M0015 *Adversarial Input Detection* — today it only detects generic low confidence, it isn't designed specifically to detect adversarial input (see "Limitations" below) |
| *(not implemented)* query rate limiting | AML.M0004 *Restrict Number of AI Model Queries* — a direct mitigation against the black-box attack (SimBA), which needs thousands of queries to converge |

### Limitations of this mapping

- The confidence `--limiar` wasn't designed as a defense against adversarial attack — it solves a different problem (a sign outside the GTSRB's 43 classes). A successful adversarial attack tends to produce a prediction with *high* confidence in the wrong class (as the tests above show: 95-99% confidence), so the threshold doesn't catch those cases. Stating this as a limitation is more honest than claiming a defense that doesn't exist.
- We haven't actually implemented AML.M0004 (rate limiting) — it's an infrastructure/API-level mitigation, out of this repository's scope, which only makes sense once the model is actually sitting behind an API serving requests.

## Robustness curve (accuracy × attack strength)

`eval_robustez.py` runs FGSM/PGD against the whole test set (or a sample, with `--amostra`) and measures accuracy for each eps value — unlike the previous attacks, which evaluate a single image. The physical patch and the black-box attack don't go into this sweep because they optimize per image (too expensive to run on all 12,630 test images); they're left with the point-wise numbers already reported in the sections above.

```bash
python -m analise.eval_robustez --modelo sinalnet.pt sinalnet_robusto.pt --eps 0 0.01 0.02 0.03 0.05 0.08 0.1 --csv robustez.csv
```

Measured result (sample of 512 images, PGD, 5 steps):

| eps | `sinalnet.pt` (normal) | `sinalnet_robusto.pt` (adv. training, 80 epochs, with eps warm-up and 70/30 mixing) |
|---|---|---|
| 0.00 | 97.5% | 93.2% |
| 0.03 | 14.3% | 65.2% |
| 0.08 | 10.4% | 51.8% |

The curve makes visual what the earlier sections only described in text: the normal model **collapses** as soon as the attack shows up (97.5% → 14.3% at eps=0.03, an 83-point drop), while the robust one loses accuracy much more gently (93.2% → 65.2%, a 28-point drop). With the hyperparameter adjustments (eps warm-up, attacking only 70% of the batch), the robust model got much closer to the normal model's clean accuracy than in the first attempt (77.5% → 93.2%), without losing robustness — both metrics improved together, it's not just a different point on the same trade-off.

## Does the defense generalize to other attacks?

Adversarial training was only trained against PGD. That raises an obvious question: does it also help against the other two attacks (physical patch, black-box), which it never saw during training? `avaliar_defesas.py` tests this, running both attacks against both checkpoints on a sample of test images.

```bash
python -m analise.avaliar_defesas --amostra 20
```

Measured result (20 images, untargeted attack):

| Attack | `sinalnet.pt` (normal) | `sinalnet_robusto.pt` (robust) | Reduction |
|---|---|---|---|
| PGD (eps=0.03) — what training uses | ~86% success | ~39% success | **-47 points** |
| Physical patch | 85% success | 50% success | **-35 points** |
| Black-box (SimBA) | 85% success | 60% success | **-25 points** |

**Finding**: the defense **generalizes partially** — it reduces the success rate of the other two attacks as well, even though it never saw either of them during training. But the reduction gets weaker the further the attack is from what was used for training: PGD (the training attack itself) drops 47 points, the physical patch (still gradient-based, but on a restricted region and with robustness to transformation) drops 35, and the black-box (which doesn't even use a gradient) drops only 25. This is consistent with the intuition that adversarial training learns to resist a specific perturbation *direction*, not any perturbation whatsoever — the more different the attack, the less the defense generalizes.

## Transfer between the two models

A question raised by the adversarial-training defense: does an adversarial example only work against the model it was generated on, or does it attack any similar model? `transferencia.py` tests this: it generates adversarials on one model (PGD, eps=0.03) and measures accuracy both on that model and on the other one.

```bash
python -m analise.transferencia --amostra 2048
```

Measured result (2,048 test images):

| Adversarial generated on → evaluated on | Accuracy |
|---|---|
| normal → normal (white-box, same model) | 11.0% |
| normal → robust (transfer) | 88.8% |
| robust → robust (white-box, same model) | 60.7% |
| robust → normal (transfer) | 64.7% |

**Finding**: transfer is low in both directions — an adversarial example made for the normal model barely affects the robust one (88.8%, close to its own clean accuracy) and vice versa (64.7%, a real drop from the normal model's ~97.5% clean accuracy, but still well above what the attack achieves on the model it was generated on). This means the robustness of an adversarially-trained model isn't "generic": it's relatively specific to that particular model. From a threat standpoint, that's good news (an attacker can't easily train a similar substitute and reuse the adversarials directly against a different system), but it also means that validating the robustness of **one** model doesn't guarantee anything about another model, even if they look similar.

## Attempt at adversarial input detection (negative result)

The AML.M0015 mitigation (*Adversarial Input Detection*) suggests detecting adversarial input before even classifying it. We tested two simple, cheap heuristics, implemented in `deteccao.py`:

1. **Instability under noise**: the hypothesis was that an adversarial image, being right next to the decision boundary (it was perturbed by the smallest distance that sufficed to cross it), would have an unstable prediction under a bit of random noise, while a clean image, further "inside" its class's region, would keep its prediction.
2. **Confidence margin** (top-1 minus top-2 probability): the hypothesis was that a successful attack would leave a small margin.

**Neither held up in testing**: comparing 8 clean test images against the same ones after PGD (eps=0.03), the disagreement rate under noise didn't separate the two groups reliably (in some cases the *clean* image was actually more unstable than the adversarial one), and the confidence margin stayed high (often ~1.0) both before and after the attack — in other words, PGD produces errors with *high* confidence, not low. This is consistent with the literature: PGD explicitly optimizes to cross the decision boundary efficiently, so the resulting point isn't "randomly" close to the boundary in every direction, only in the specific direction of the gradient — random noise (which probes generic directions) doesn't necessarily reveal that.

We keep `deteccao.py` in the repository as evidence of this experiment, not as a defense that works. The finding itself is relevant to the project: it reinforces why *model hardening* (adversarial training) is the most reliable mitigation we tested, and why cheap, generic detection isn't a trivial substitute for it.
