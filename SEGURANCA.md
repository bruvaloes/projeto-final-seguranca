# Análise de segurança

Este documento trata a SinalNet como o alvo de uma análise de risco, não como um projeto de machine learning: o objetivo é entender o que um atacante pode fazer contra o modelo, o quão viável isso é, e o que mitiga cada risco. O código (`attack.py`, `train.py`, `predict.py`) é a ferramenta usada para investigar essas perguntas, não o produto final do trabalho.

## Modelo de ameaça e risco

| | |
|---|---|
| **Ativo** | A decisão de classificação da SinalNet — qual placa o sistema "acha" que está vendo. |
| **Ameaça** | Um atacante que consegue alterar a imagem de entrada (digitalmente ou perturbando a placa física) para forçar uma classificação diferente da correta. |
| **Vulnerabilidade** | O modelo é uma rede neural treinada por gradiente: pequenas perturbações alinhadas com o gradiente da perda bastam para mudar a predição, mesmo quando a imagem continua, pra um humano, claramente a mesma placa (`attack.py`, FGSM e PGD). |
| **Viabilidade** | Alta no modelo sem defesa: com acesso aos pesos (white-box), PGD já muda a predição com perturbações pequenas (eps ~0,03-0,08 em pixels normalizados entre 0 e 1) em poucas iterações. Discutimos a viabilidade com acesso mais restrito (sem gradiente, sem acesso físico) nas seções seguintes deste documento. |
| **Impacto** | Depende de onde a decisão é consumida — ver [Cenário de ameaça](#cenário-de-ameaça) abaixo. Em geral, qualquer sistema que decide uma ação de segurança (parar, não ultrapassar, respeitar limite de velocidade) com base nessa classificação propaga o erro adiante. |
| **Mitigação** | *Adversarial training* (`train.py --adv-treino`) reduz a viabilidade do ataque, ao custo de acurácia limpa — ver a seção "Adversarial training" do [README](README.md). Isso é redução de risco, não eliminação: a Seção de resultados do README mostra que o modelo robusto ainda erra sob ataque, só que menos. |

### Por que isso é diferente de um bug comum

Um bug de software tradicional (buffer overflow, injeção de SQL) tem uma correção que o elimina. Robustez adversarial não funciona assim: não existe "patch" que torne a rede imune a qualquer perturbação dentro do orçamento do atacante — o máximo que se consegue é deslocar a fronteira de decisão para que perturbações do mesmo tamanho exijam mais esforço computacional do atacante. Por isso a métrica certa não é "o ataque funciona? sim/não", e sim "quanto custa (em eps, em consultas, em acesso) montar o ataque, e o que isso custa pro modelo robusto em troca".

## Cenário de ameaça

Pra discutir impacto de forma concreta, imaginamos a SinalNet como o estágio de reconhecimento de placas de um pipeline de ADAS (sistema avançado de assistência ao motorista) ou carro autônomo: câmera → pré-processamento → **classificador de placa** → módulo de decisão (frear, respeitar limite de velocidade, não ultrapassar). A classificação errada não fica isolada: ela se propaga para uma ação do veículo.

Duas superfícies de ataque bem diferentes, com atacantes de capacidade bem diferente:

### 1. Superfície digital — alguém no caminho da imagem

**Atacante**: tem acesso ao pipeline de dados entre a câmera e o classificador (ex.: um componente de software comprometido, ou um ataque na própria câmera/driver). Pode alterar os pixels antes de chegarem ao modelo.

**Capacidade**: se também tiver acesso aos pesos do modelo (white-box — o caso mais forte, e o que `attack.py` assume com FGSM/PGD), a perturbação necessária é pequena (eps ~0,03-0,08) e rápida de calcular. Isso é o cenário mais fácil de simular e o que já implementamos.

**Realismo**: alto *se* o atacante já comprometeu algo no pipeline — mas nesse ponto ele já tem bastante acesso ao sistema, então a classificação de placa pode não ser o elo mais fácil de atacar. Esse ataque é mais relevante como estudo da robustez do modelo em si do que como vetor de intrusão isolado.

### 2. Superfície física — alguém que só tem acesso à placa

**Atacante**: não tem acesso a nenhum software do carro, só pode alterar o mundo físico — colar um adesivo na placa, por exemplo. Não precisa comprometer nada digital.

**Capacidade**: mais restrita e mais interessante do ponto de vista de segurança, porque é um vetor de ataque real e barato (qualquer pessoa pode se aproximar de uma placa). Mas é tecnicamente mais difícil: a perturbação precisa sobreviver a variações de ângulo, distância, iluminação e à captura da câmera — uma perturbação de pixel calculada para uma única foto (como o FGSM/PGD padrão) geralmente não sobrevive a essas transformações. Isso é tratado na seção "Ataque físico" abaixo.

**Realismo**: esse é o cenário citado na literatura de segurança de ML (ex.: adesivos em placas de Pare) como o mais preocupante na prática, justamente porque não exige nenhum acesso digital — só proximidade física com a placa.

### Qual importa mais pro trabalho

A superfície física é a que mais justifica chamar isso de "projeto de segurança": não depende de uma premissa forte (acesso aos pesos do modelo) para ser um risco real. A superfície digital é mais fácil de demonstrar e quantificar, e serve de base técnica (é o que os ataques FGSM/PGD já implementados exploram), mas seu impacto prático depende de um atacante que já tem outro ponto de entrada no sistema.

## Ataque físico (patch adversarial)

`patch_attack.py` implementa a superfície física descrita acima: em vez de perturbar a imagem inteira (FGSM/PGD), otimiza um pequeno adesivo quadrado — um patch — colado num canto fixo da placa, que precisa continuar enganando o modelo mesmo sob variações de ângulo, distância e iluminação da foto.

```bash
python patch_attack.py placa.jpg                                   # nao-direcionado
python patch_attack.py placa.jpg --classe 14 --tamanho 24 --passos 600  # forca "Pare"
```

A técnica é **Expectation over Transformation** (EoT, de Athalye et al., "Synthesizing Robust Adversarial Examples"): a cada passo do treino do patch, em vez de calcular o gradiente numa única versão da imagem, tiramos a média do gradiente sobre `--amostras-por-passo` versões com rotação, translação, escala e brilho aleatórios (`_compor` em `patch_attack.py`). Isso é o que diferencia esse ataque do FGSM/PGD: o patch não é otimizado pra uma foto específica, é otimizado pra sobreviver a uma distribuição de condições de captura — o requisito real de um adesivo físico.

**Achado relevante pro trabalho**: o ataque não-direcionado (só forçar qualquer erro) converge rápido e com alta confiança mesmo com uma faixa de EoT pequena. O ataque **direcionado** (forçar uma classe específica, ex. fazer o modelo ver "Pare" em qualquer outra placa) precisa de um patch maior e mais passos para convergir de forma confiável — com `--tamanho 16` e EoT mais larga ele não converge bem; com `--tamanho 24` (um quarto da área da imagem) e 600 passos, chega a 98,6% de confiança na classe forçada. Ou seja: **é consideravelmente mais fácil um atacante causar um erro qualquer do que escolher exatamente qual erro causar** — uma assimetria de risco que vale citar na análise: mesmo um atacante sem controle fino ainda consegue negar o reconhecimento correto da placa.

## Ataque black-box (baseado em consultas)

Todos os ataques anteriores (FGSM, PGD, o patch) assumem acesso aos pesos do modelo para calcular o gradiente — **white-box**. Um atacante real contra um sistema implantado (por trás de uma API, ou de um binário que ele não conseguiu extrair os pesos) normalmente não tem isso. `blackbox_attack.py` implementa o **SimBA** (Guo et al., "Simple Black-box Adversarial Attacks"): a cada passo, tenta perturbar um pixel aleatório pra cima ou pra baixo e só olha se a probabilidade da classe de interesse melhorou — nunca usa `.backward()` nem acessa gradiente, só consulta a saída do modelo (`model(x)`), que é exatamente o que um atacante com apenas acesso de API conseguiria fazer.

```bash
python blackbox_attack.py placa.jpg                                    # nao-direcionado
python blackbox_attack.py placa.jpg --classe 14 --passos 6912 --step 0.3  # forca "Pare"
```

**Achado relevante pro trabalho**: o ataque funciona (testado: não-direcionado chega a 95%+ de confiança numa classe errada, direcionado chega a 99,96% forçando "Pare"), mas custa muito mais **consultas ao modelo** que teria custado chamadas de gradiente num ataque white-box — no teste, o não-direcionado usou ~4.700 consultas e o direcionado precisou escanear a imagem inteira, ~11.400 consultas, pra convergir bem. Isso é o ponto central de um ataque black-box: ele é viável, mas tem um **custo mensurável e observável** (número de chamadas ao modelo) que não existe no ataque digital white-box nem no ataque físico. Esse número de consultas é justamente o que a mitigação "restringir o número de consultas ao modelo" (ver [mapeamento MITRE ATLAS](#mapeamento-mitre-atlas)) tenta limitar — um sistema que detecta e bloqueia um cliente fazendo milhares de consultas na mesma imagem inviabiliza esse ataque específico, mesmo sem tocar no modelo.
