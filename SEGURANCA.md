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
