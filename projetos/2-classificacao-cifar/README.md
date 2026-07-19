# Projeto 2 — Classificação CIFAR-10

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar imagens coloridas** em 10 categorias de objetos e animais (avião, automóvel, pássaro, gato, cervo, cachorro, sapo, cavalo, navio, caminhão), e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

Este projeto tem uma diferença importante em relação a uma classificação de dígitos: as imagens são **coloridas (RGB)** e visualmente mais complexas, o que torna a tarefa de classificação genuinamente mais difícil — por isso **data augmentation** é um requisito obrigatório aqui, não opcional.

## 🎯 Conjunto de Dados

Dataset **CIFAR-10**, disponível diretamente via `tf.keras.datasets.cifar10` (não é necessário download manual). 60.000 imagens 32x32 coloridas, 10 classes.

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset CIFAR-10 via TensorFlow
- Split explícito treino/validação
- **Data augmentation** aplicada ao conjunto de treino, usando camadas do Keras
  (ex: `RandomFlip("horizontal")`, `RandomRotation`, `RandomZoom`) incorporadas ao
  modelo ou ao pipeline de treino
- Construção de uma CNN com 3-4 blocos convolucionais (`Conv2D` + `BatchNormalization`
  + `MaxPooling2D`) seguida de `Dropout`
- Treinamento com **early stopping** baseado na perda de validação
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

> 💡 Se você aplicar a augmentation de outra forma (ex: pré-processamento manual em
> `tf.data`), tudo bem — apenas descreva isso claramente no relatório, já que a
> correção automática busca primeiro por camadas de augmentation no próprio modelo.

> 💡 CIFAR-10 é mais difícil que MNIST/Fashion-MNIST para uma CNN simples treinada
> rapidamente em CPU — não se preocupe se a acurácia ficar bem abaixo de 90%. O
> importante é o pipeline completo funcionar corretamente.

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/2-classificacao-cifar/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 32x32, 3 canais (RGB), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 25-30, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Generalização** — uso adequado de data augmentation
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:Raioni Felix de Sousa**

### 1️⃣ Resumo da Arquitetura do Modelo

A CNN é composta por 3 blocos convolucionais (Conv2D + BatchNormalization + MaxPooling2D), com número de filtros crescente entre os blocos (32 → 64 → 128), seguidos de uma camada Flatten, Dropout (0.4), uma camada Dense de 128 neurônios com ativação ReLU, um segundo Dropout (0.3) e a camada de saída Dense com 10 neurônios e ativação softmax.
A estratégia de data augmentation foi implementada como camadas do Keras (RandomFlip("horizontal"), RandomRotation, RandomZoom) incorporadas diretamente ao modelo, logo após a camada de entrada. Por estarem dentro do próprio modelo, essas camadas atuam apenas durante o treinamento (training=True) e são automaticamente desativadas durante avaliação e inferência, sem necessidade de lógica adicional.

### 2️⃣ Bibliotecas Utilizadas

TensorFlow (versão instalada via requirements.txt, >=2.12 — 2.21.0)
NumPy

### 3️⃣ Técnica de Otimização do Modelo

Foi utilizada a Dynamic Range Quantization, aplicada via converter.optimizations = [tf.lite.Optimize.DEFAULT] no TFLiteConverter. Essa técnica converte os pesos do modelo de float32 para int8, mantendo as ativações calculadas em ponto flutuante durante a inferência (a faixa de valores é calculada dinamicamente em tempo de execução). A vantagem é reduzir significativamente o tamanho do modelo.

### 4️⃣ Resultados Obtidos


Acurácia de validação final: 0.7416 (74.16%)
Tamanho do model.h5: 4.2 MB
Tamanho do model.tflite: 365 KB
Redução de tamanho obtida com a otimização foi de aproximadamente 11.5 vezes.

### 5️⃣ Comentários Adicionais (Opcional)

Durante o treinamento um dos estranhamentos foi a queda do val_accuracy, mas pesquisando foi possível observar que, nas épocas finais, a val_accuracy ocila (chegando a cair levemente entre épocas consecutivas) enquanto a accuracy de treino seguia subindo de forma consistente — mostrando um sinal característico de que o modelo estava se aproximando do limite de generalização para essa arquitetura, um comportamento inicial de overfitting. O EarlyStopping, monitorando val_loss com patience=5 e restore_best_weights=True, garantiu que o modelo final salvo correspondesse aos pesos da melhor época observada, e não necessariamente aos da última.
Outro ponto relevante foi o resultado da otimização: a redução de ~11.5x no tamanho do arquivo (de 4.2 MB para 365 KB) ficou acima da faixa tipicamente esperada para Dynamic Range Quantization (~3-4x), o que sugere que grande parte dos parâmetros do modelo está concentrada nas camadas densas (Dense), que se beneficiam fortemente da quantização de pesos.
O treinamento foi realizado inteiramente em CPU, conforme restrição do desafio, com cada época levando entre 260s e 341s.

### 6️⃣ Exemplo de Inferência

Amostra 1: predito=cat | real=cat
Amostra 2: predito=ship | real=ship
Amostra 3: predito=ship | real=ship
Amostra 4: predito=airplane | real=airplane
Amostra 5: predito=frog | real=frog
Todas as 5 amostras testadas foram classificadas corretamente. Vale destacar, porém, que esse resultado (5/5) não deve ser interpretado como acurácia de 100% do modelo — trata-se de uma amostra muito pequena para ter significância estatística. A métrica confiável para avaliar o desempenho real do modelo é a acurácia de validação de 74.16%, medida sobre o conjunto de validação completo. 