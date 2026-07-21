# Projeto 2 — Classificação CIFAR-10

## 📝 Relatório do Candidato

👤 **Nome Completo:Raioni Felix de Sousa**

### 1️⃣ Resumo da Arquitetura do Modelo

A CNN é composta por 3 blocos convolucionais (Conv2D + BatchNormalization + MaxPooling2D), com número de filtros crescente entre os blocos (32 → 64 → 128), seguidos de uma camada Flatten, Dropout (0.4), uma camada Dense de 128 neurônios com ativação ReLU, um segundo Dropout (0.3) e a camada de saída Dense com 10 neurônios e ativação softmax.

A estratégia de data augmentation foi implementada como camadas do Keras (RandomFlip("horizontal"), RandomRotation, RandomZoom) incorporadas diretamente ao modelo, logo após a camada de entrada. Por estarem dentro do próprio modelo, essas camadas atuam apenas durante o treinamento (training=True) e são automaticamente desativadas durante avaliação e inferência, sem necessidade de lógica adicional.

### 2️⃣ Bibliotecas Utilizadas

TensorFlow ( Version: 2.16.1)
tf-keras~=2.16
NumPy

### 3️⃣ Técnica de OtimizSação do Modelo

Foi utilizada a Dynamic Range Quantization, aplicada via converter.optimizations = [tf.lite.Optimize.DEFAULT] no TFLiteConverter. Essa técnica converte os pesos do modelo de float32 para int8, mantendo as ativações calculadas em ponto flutuante durante a inferência (a faixa de valores é calculada dinamicamente em tempo de execução). A vantagem é reduzir significativamente o tamanho do modelo.

### 4️⃣ Resultados Obtidos


Acurácia de validação final: 0.7860
Tamanho do model.h5: 4.2 MB
Tamanho do model.tflite: 362 KB
Redução de tamanho obtida com a otimização foi de aproximadamente 11.5 vezes.

### 5️⃣ Comentários Adicionais (Opcional)

Durante o treinamento no Google Colab( usando Python 3.12.13) um dos estranhamentos foi a queda do val_accuracy, mas pesquisando foi possível observar que, nas épocas finais, a val_accuracy ocila (chegando a cair levemente entre épocas consecutivas) enquanto a accuracy de treino seguia subindo de forma consistente — mostrando um sinal característico de que o modelo estava se aproximando do limite de generalização para essa arquitetura, um comportamento inicial de overfitting. O EarlyStopping, monitorando val_loss com patience=5 e restore_best_weights=True, garantiu que o modelo final salvo correspondesse aos pesos da melhor época observada, e não necessariamente aos da última.

Outro ponto relevante foi o resultado da otimização: a redução de ~11.5x no tamanho do arquivo (de 4.2 MB para 365 KB) ficou acima da faixa tipicamente esperada para Dynamic Range Quantization (~3-4x), o que sugere que grande parte dos parâmetros do modelo está concentrada nas camadas densas (Dense), que se beneficiam fortemente da quantização de pesos.
O treinamento foi realizado inteiramente em CPU, conforme restrição do desafio.

### 6️⃣ Exemplo de Inferência

Amostra 1: predito=cat | real=cat
Amostra 2: predito=ship | real=ship
Amostra 3: predito=ship | real=ship
Amostra 4: predito=airplane | real=airplane
Amostra 5: predito=frog | real=frog
Todas as 5 amostras testadas foram classificadas corretamente. Vale destacar, porém, que esse resultado (5/5) não deve ser interpretado como acurácia de 100% do modelo — trata-se de uma amostra muito pequena para ter significância estatística. A métrica confiável para avaliar o desempenho real do modelo é a acurácia de validação de 0.7860, medida sobre o conjunto de validação completo. 