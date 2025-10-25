# Introdução à Ciência de Dados e Aprendizado de Máquina

Métodos de ciência de dados são utilizados em diversos setores para gerar valor para os negócios.  
Neste curso, abordaremos os seguintes objetivos de aprendizado:

- Explorar alguns métodos comuns de ciência de dados;
- Discutir diferentes casos de uso desses métodos;
- Explicar como avaliar o desempenho desses métodos;
- Definir o que é aprendizado de máquina (machine learning, ou ML);
- Cobrir algumas tarefas comuns de aprendizado de máquina, incluindo **agrupamento (clustering)**, **classificação (classification)**, **regressão (regression)** e **engenharia de atributos (feature engineering)**.

---

## Introdução ao Machine Learning

O **aprendizado de máquina** é um tipo de **inteligência artificial (IA)** que utiliza algoritmos para encontrar **padrões em grandes volumes de dados** e prever resultados futuros com **mínima intervenção humana**.  
Esses algoritmos usam **dados históricos** para fazer previsões e geralmente são classificados como **supervisionados** ou **não supervisionados**.

O aprendizado de máquina alimenta muitos serviços modernos, como:

- **Sistemas de recomendação** (Netflix, Amazon);
- **Motores de busca** (Google);
- **Feeds de redes sociais** (Facebook, Twitter/X);
- **Assistentes de voz** (Siri, Alexa);
- **Manutenção preditiva** em transporte e logística;
- **Análise de registros médicos** em saúde;
- **Detecção de fraudes** em instituições financeiras;
- **Análise de clientes** em vendas e marketing (churn rate, lifetime value).

---

## Aprendizado Supervisionado

O aprendizado supervisionado **usa dados rotulados** (ou seja, com entradas e saídas conhecidas) para treinar o modelo.  
A máquina aprende a **relação entre as variáveis de entrada (X)** e as **variáveis de saída (Y)**, para poder prever novos valores.

### Exemplos

- Classificar e-mails como **spam** ou **não spam**;  
- Diagnóstico médico com base em imagens;  
- Reconhecimento facial ou de voz;  
- Previsão de mercado, clima, ou PIB.

### Técnicas comuns

- **Classificação (Classification)**: divide dados em categorias (ex: spam/não spam);  
- **Regressão (Regression)**: prevê valores numéricos (ex: preço de casas);  
- **Processamento de Linguagem Natural (NLP)**: reconhecimento de fala, tradução, análise de sentimentos;  
- **Deep Learning**: reconhecimento de objetos e geração de texto.

---

## Aprendizado Não Supervisionado

O aprendizado não supervisionado **trabalha com dados sem rótulos**.  
O objetivo é **descobrir padrões ocultos**, **estruturas** ou **relações** dentro dos dados.

### Exemplo didático

Se uma criança recebe um prato com **bananas e laranjas** e as separa por **cor e formato**, isso é equivalente a **clustering**.

### Técnicas principais

- **Clustering (Agrupamento)**: agrupa dados semelhantes (ex: clientes com hábitos parecidos);  
- **Redução de dimensionalidade**: simplifica dados complexos;  
- **Modelagem de tópicos**: identifica temas em grandes coleções de textos;  
- **Detecção de anomalias**: identifica padrões incomuns (ex: fraudes).

### Usos práticos

- **Segmentação de mercado**;  
- **Detecção de fraudes**;  
- **Recomendações personalizadas**;  
- **Controle de qualidade** na manufatura.

---

## Engenharia de Atributos (Feature Engineering)

Processo de criar e selecionar as **melhores variáveis** (features) para melhorar o desempenho do modelo.

### Técnicas

- **Criação de atributos**: gerar novos dados a partir de conhecimento de domínio;  
- **Transformação**: converter atributos para formatos mais úteis (ex: log, normalização);  
- **Escalonamento**: padronizar faixas de valores;  
- **Seleção**: escolher os atributos mais relevantes;  
- **Extração**: reduzir a quantidade de dados mantendo as informações essenciais.

### Linguagens e bibliotecas

- **Python**: Pandas, NumPy, scikit-learn, Featuretools;  
- **R**: dplyr, tidyr, caret.

---

## Clustering (Agrupamento)

O **clustering** é um método de aprendizado **não supervisionado** usado para **encontrar similaridades** entre pontos de dados e agrupá-los.  
Cada grupo é chamado de **cluster**.

### Vantagens

- Descobre padrões ocultos;  
- Ajuda na visualização de dados;  
- Útil quando não há rótulos disponíveis.

### Desvantagens

- Pode fazer suposições incorretas sobre o formato dos dados;  
- Difícil de interpretar;  
- Alguns algoritmos não escalam bem para grandes volumes de dados.

### Exemplo real

Em 1997, pesquisadores usaram clustering para **dividir clientes de cartão de crédito** em cinco grupos com base em comportamento de pagamento, ajudando campanhas de marketing mais precisas.

---

## Classificação (Classification)

Tipo de aprendizado supervisionado que **atribui novos dados a classes conhecidas**.

### Usos

- **Filtragem de spam**;  
- **Diagnóstico médico**;  
- **Reconhecimento de imagem**;  
- **Veículos autônomos**.

### Modelos comuns

- **Regressão logística**;  
- **Máquinas de vetores de suporte (SVM)**;  
- **Árvores de decisão / Random Forest / XGBoost**;  
- **K-vizinhos mais próximos (KNN)**.

### Avaliação

- **Matriz de confusão** (verdadeiro/falso positivo/negativo);  
- **Curva ROC** e **AUC** (área sob a curva).

---

## Regressão (Regression)

A **regressão** prevê valores numéricos com base em outras variáveis.  
Exemplo: prever o **preço de uma casa** com base em **tamanho, localização e idade**.

### Tipos

- **Regressão linear simples**: uma variável independente;  
- **Regressão múltipla**: várias variáveis independentes.

### Métricas de avaliação

- **R²** (variância explicada);  
- **RMSE** (erro quadrático médio);  
- **MAE** (erro absoluto médio).

### Cuidados

- **Multicolinearidade** (variáveis independentes correlacionadas);  
- **Autocorrelação**;  
- **Heterocedasticidade** (variância desigual dos erros).

---

## Desafios Comuns em Machine Learning

1. **Poucos dados de treino** → risco de *overfitting*  
   🟢 Solução: coletar mais dados ou gerar dados sintéticos (ex: SMOTE).

2. **Dados não representativos** → *sampling bias*  
   🟢 Solução: melhorar a amostragem e equilibrar viés e variância.

3. **Baixa qualidade dos dados** → erros, duplicatas, valores ausentes  
   🟢 Solução: limpeza e padronização dos dados.

4. **Atributos irrelevantes**  
   🟢 Solução: aplicar **feature engineering**, seleção e extração de atributos.

---

## Conclusão

Nesta seção, aprendemos:

- O que é **Machine Learning** e suas aplicações;  
- Os métodos fundamentais: **Clustering**, **Classificação** e **Regressão**;  
- Como usar **Feature Engineering** para melhorar modelos;  
- E como enfrentar desafios comuns em projetos de aprendizado de máquina.

Em seguida, o curso abordaria **métodos avançados de ciência de dados**.
