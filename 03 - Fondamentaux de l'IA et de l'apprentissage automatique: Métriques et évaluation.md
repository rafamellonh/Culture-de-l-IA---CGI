
[![Ouvir Áudio](https://img.shields.io/badge/▶️%20Ouvir%20Áudio-4285F4?style=for-the-badge)](https://drive.google.com/file/d/12X4kTm9hKT-kgwx0yvNBHzh50s_Nq9qr/view?usp=sharing)
[![Assistir Aula](https://img.shields.io/badge/▶️%20Assistir%20Aula-4285F4?style=for-the-badge)](https://drive.google.com/file/d/1sbNhFE2RRSKCJQzh27fyfdb4enf0uFnR/view?usp=drive_link)


# Fundamentos de IA e ML: Métricas e Avaliação

Entender a avaliação de modelos é essencial para tomar decisões
confiáveis, precisas e éticas ao usar inteligência artificial (IA) e
aprendizado de máquina (ML) em cenários práticos.\
Neste curso, você vai explorar em profundidade a avaliação e a
interpretabilidade de modelos de IA/ML, ganhando uma base sólida nesses
componentes essenciais para fazer IA/ML funcionar de forma eficaz na sua
organização.

O curso foca nos conceitos e métricas principais necessários para
avaliar o quão bem os modelos estão performando.\
Ao concluir este curso, você estará bem preparado para tomar decisões
informadas e maximizar o potencial de IA/ML dentro da sua organização.

------------------------------------------------------------------------

## 1. Visão geral do curso

Neste vídeo, vamos descobrir os principais conceitos abordados neste
curso.

Métricas e avaliação em inteligência artificial e aprendizado de
máquina.\
Neste curso, vamos discutir:

-   Métricas de avaliação para modelos de classificação e de regressão\
-   Desafios de avaliação, como sobreajuste (overfitting), viés e
    desbalanceamento de classes\
-   Técnicas de mitigação para esses desafios\
-   Boas práticas na hora de avaliar seu modelo\
-   Tendências emergentes em métricas e avaliação em IA/ML

------------------------------------------------------------------------

## 2. Métricas de Avaliação para Classificação

Depois de concluir esta parte, você será capaz de **definir métricas
comuns de avaliação para classificação** e entender **por que elas
importam dependendo do problema**.

Métricas de avaliação são usadas para medir o desempenho de modelos de
machine learning. Elas:

-   São quantitativas\
-   Ajudam a comparar modelos e algoritmos\
-   Variam conforme o tipo de modelo (classificação vs regressão)

### Classificação binária

Modelos de classificação binária tentam decidir se um ponto de dados
pertence ao grupo alvo ou não.\
Exemplo: detectar se um e-mail é spam ou não.

Para analisar esse tipo de modelo, usamos estes conceitos:

-   **Verdadeiro Positivo (TP):** modelo disse "spam", e era spam\
-   **Verdadeiro Negativo (TN):** modelo disse "não é spam", e realmente
    não era spam\
-   **Falso Positivo (FP):** modelo disse "spam", mas não era spam\
-   **Falso Negativo (FN):** modelo disse "não é spam", mas era spam

### Métricas principais

#### Acurácia

Indica quão correta é a previsão geral do modelo.\
"Com que frequência o modelo acerta o rótulo?"

#### Precisão

Mede a qualidade das previsões positivas.\
"Dos itens que o modelo marcou como positivos, quantos eram realmente
positivos?"

#### Recall

Mede a capacidade de encontrar todos os itens positivos.\
"Dos itens que eram realmente positivos, quantos o modelo encontrou?"

#### F1-score

É a média harmônica entre precisão e recall:

    F1 = 2 × (Precisão × Recall) / (Precisão + Recall)

### Problemas multiclasse

Para classificação com 3+ classes (ex.: maçã, banana, laranja), usamos
métricas como:

-   Entropia cruzada categórica\
-   Macro F1\
-   Micro F1\
-   Top-k accuracy

------------------------------------------------------------------------

## 3. Métricas de Avaliação para Regressão

Problemas de regressão tentam prever valores numéricos contínuos.

### Principais métricas

-   **MAE:** erro médio absoluto\
-   **MSE:** erro quadrático médio\
-   **RMSE:** raiz do erro quadrático médio\
-   **R²:** proporção da variação explicada pelo modelo

Cada uma responde a uma pergunta diferente, e a escolha depende do
contexto do problema.

------------------------------------------------------------------------

## 4. Overfitting (Sobreajuste)

Overfitting é quando o modelo aprende demais os dados de treino,
incluindo ruídos e detalhes aleatórios.\
Isso faz com que ele funcione bem no treino e mal em dados novos.

### Como mitigar

-   **Validação cruzada**\
-   **Seleção de variáveis**\
-   **Regularização (Ridge, Lasso)**\
-   **Mais dados de treino**

------------------------------------------------------------------------

## 5. Matrizes de Confusão

A matriz de confusão mostra onde o modelo acerta e onde erra.

  Real / Previsto   Positivo           Negativo
  ----------------- ------------------ -------------------
  **Positivo**      TP (acerto)        FN (erro tipo II)
  **Negativo**      FP (erro tipo I)   TN (acerto)

Permite calcular **acurácia**, **precisão**, **recall** e **F1-score**.

------------------------------------------------------------------------

## 6. ROC e AUC

-   **ROC:** curva que plota TPR (recall) vs FPR (falsos positivos)\
-   **AUC:** área sob a curva ROC (0 a 1)
    -   Quanto mais próximo de 1, melhor o modelo separa as classes

------------------------------------------------------------------------

## 7. Viés na Avaliação de Modelos

O viés surge quando o modelo ou os dados favorecem injustamente certos
grupos.\
Fontes comuns:

-   Viés de amostragem\
-   Viés algorítmico\
-   Viés histórico\
-   Viés de avaliação

### Mitigação

-   Usar dados representativos\
-   Avaliar desempenho por grupo demográfico\
-   Usar métricas de justiça\
-   Tornar o processo transparente

------------------------------------------------------------------------

## 8. Falsos Positivos e Falsos Negativos na Tomada de Decisão

-   **Falso positivo:** falso alarme (ex.: marcar inocente como
    culpado)\
-   **Falso negativo:** falha em detectar (ex.: deixar passar uma
    fraude)

A escolha entre minimizar um ou outro depende do contexto.

### Exemplo:

-   Saúde → reduzir falsos negativos (recall)\
-   Segurança → reduzir falsos positivos (precisão)

------------------------------------------------------------------------

## 9. Validação Cruzada

Técnica usada para testar a capacidade de generalização de um modelo.\
Divide os dados em várias partes e testa o modelo em diferentes
combinações.

Tipos: - Hold-out\
- K-fold\
- Leave-one-out\
- Leave-p-out

Benefícios: - Evita overfitting\
- Melhora a estimativa real de performance

------------------------------------------------------------------------

## 10. Desbalanceamento de Classes

Quando uma classe aparece muito mais que outra, o modelo tende a ignorar
a menor.

### Soluções

-   Reamostragem (SMOTE, oversampling, undersampling)\
-   Pesos de classe\
-   Métricas adequadas (F1, recall, precisão)\
-   Coletar mais dados

------------------------------------------------------------------------

## 11. Interpretabilidade em IA e ML

Interpretabilidade = entender por que o modelo tomou determinada
decisão.

### Importância

-   Transparência e confiança\
-   Responsabilidade e ética\
-   Detecção de viés\
-   Explicações em domínios críticos (saúde, finanças, RH)

### Técnicas

-   Importância de variáveis\
-   SHAP\
-   LIME\
-   Mapas de saliência\
-   Anchors

------------------------------------------------------------------------

## 12. Acompanhamento de Desempenho (Performance Tracking)

Monitorar continuamente os modelos é essencial para garantir:

-   Alinhamento com objetivos de negócio\
-   Qualidade em produção\
-   Adaptação a mudanças\
-   Conformidade ética e regulatória

### Ferramentas comuns

Prometheus, Grafana, ELK, Datadog, TensorBoard, AutoML, scripts
customizados.

------------------------------------------------------------------------

## 13. Tendências Emergentes em Avaliação de IA/ML

### Principais tendências

-   **IA Explicável (XAI):** modelos mais transparentes e
    compreensíveis\
-   **Auditoria de Justiça (Fairness Auditing):** mitigação de viés
    algorítmico\
-   **Automação de testes e avaliação:** aumento de escala e precisão\
-   **Monitoramento contínuo:** revisão ética e técnica constante\
-   **Integração de ética:** alinhamento com valores organizacionais e
    leis

------------------------------------------------------------------------

## 14. Resumo do Curso

Neste curso, foram abordados:

-   Métricas de avaliação\
-   Overfitting\
-   Matriz de confusão\
-   ROC e AUC\
-   Viés e justiça\
-   Falsos positivos/negativos\
-   Validação cruzada\
-   Desbalanceamento de classes\
-   Interpretabilidade\
-   Monitoramento e tendências emergentes

**Parabéns!**
