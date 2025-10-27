https://notebooklm.google.com/notebook/64843e90-a9f7-45db-abb8-0623cadfd16a?artifactId=2e8797e7-649a-4688-b3c9-401a9bb96ded

[![Ouvir Áudio](https://img.shields.io/badge/🎧%20Ouvir%20Áudio-4285F4?style=for-the-badge)](https://drive.google.com/file/d/1UBaZ4tfs-tEfnvFdL5P82B29J5g9YK0t/view?usp=sharing)


[![Assistir Aula](https://img.shields.io/badge/▶️%20Assistir%20Aula-4285F4?style=for-the-badge)](https://drive.google.com/file/d/1rusx8r-6XpZlpTGH9gTT1psh97mftrIl/view?usp=sharing)



# Fundamentos de IA e ML: Métodos Fundamentais de Ciência de Dados

Métodos de ciência de dados são usados em vários setores para gerar valor para os negócios. Machine Learning (ML, ou aprendizado de máquina) é um método de ciência de dados que usa algoritmos de previsão para encontrar padrões em grandes quantidades de dados, permitindo que máquinas prevejam resultados futuros e tomem decisões com intervenção humana mínima.

Este documento apresenta métodos fundamentais de uso de aprendizado de máquina.

Você vai ver o que é machine learning, como ele é categorizado e alguns casos de uso de aprendizado supervisionado e não supervisionado. Depois, verá engenharia de atributos (feature engineering) e como isso impacta a performance do modelo. Em seguida, vamos focar em tipos comuns de tarefas de ML como clusterização, classificação e regressão linear simples e múltipla. No final, você verá desafios comuns em ML e como lidar com eles.

Após a leitura, você será capaz de definir o que é machine learning e descrever métodos para usá-lo.

---

## Índice

1. Visão geral do curso  
2. Machine Learning (ML)  
3. Engenharia de atributos (Feature Engineering)  
4. Clusterização  
5. Avaliação da acurácia de algoritmos de clusterização  
6. Classificação  
7. Avaliação da acurácia de modelos de classificação  
8. Regressão  
9. Regressão linear simples  
10. Regressão linear múltipla  
11. Desafios em Machine Learning  
12. Resumo do curso  

---

## 1. Visão geral do curso

Nesta parte:
- Métodos comuns de ciência de dados  
- Casos de uso desses métodos  
- Como avaliar a performance  
- O que é machine learning (ML)  
- Tarefas centrais de ML: clusterização, classificação, regressão e engenharia de atributos  

Métodos de ciência de dados são usados em diferentes indústrias para gerar valor. Este conteúdo apresenta esses métodos e como avaliá-los.

---

## 2. Machine Learning (ML)

Nesta parte, você vai:
- Identificar casos de uso de machine learning  
- Diferenciar aprendizado supervisionado e não supervisionado  

Machine learning é um tipo de inteligência artificial (IA) que usa algoritmos para encontrar padrões em grandes quantidades de dados e prever resultados futuros com mínima intervenção humana.

Os algoritmos de ML usam dados históricos para fazer previsões. A maior parte de ML é classificada como **aprendizado supervisionado** ou **aprendizado não supervisionado**.

### Casos de uso reais de ML

Machine learning está por trás de muitos serviços do dia a dia:

- Sistemas de recomendação (streaming, e-commerce)
- Recomendações de produtos em lojas online
- Motores de busca
- Organização do feed em redes sociais
- Assistentes de voz

Outros exemplos:
- Transporte e logística: manutenção preditiva  
- Saúde: identificar grupos de pacientes para tratamento personalizado usando técnicas como processamento de linguagem natural  
- Finanças: detecção de fraude usando análise de padrões  
- Vendas e marketing: prever churn (clientes que vão sair) e valor de vida do cliente (lifetime value)

### Aprendizado supervisionado

Aprendizado supervisionado treina máquinas usando dados rotulados para que elas possam fazer previsões sobre novos dados desconhecidos.

- Você tem variáveis de entrada (X) e uma variável de saída (Y)  
- Você treina um algoritmo para aprender a função que liga X a Y  
- Objetivo: dado um novo X, prever Y  

"Dado rotulado" significa que cada exemplo vem com a resposta correta (por exemplo, "spam" ou "não spam").

Exemplos de tarefas de aprendizado supervisionado:
- Classificação (classificação de imagens, filtro de spam, detecção de fraude)
- Regressão (previsão de clima, previsão de PIB, previsão de mercado financeiro)
- Processamento de linguagem natural (reconhecimento de fala, análise de sentimento, chatbots, tradução)
- Sistemas de recomendação
- Deep learning (geração de texto, detecção de objetos, reconhecimento facial)

Exemplo:
- Um classificador de e-mail aprende quais mensagens são spam comparando os atributos delas com e-mails que um humano já marcou como spam.

Outros exemplos:
- Diagnóstico médico a partir de imagens
- Marcação automática de fotos
- Reconhecimento de pedestres em carros autônomos

### Aprendizado não supervisionado

Aprendizado não supervisionado treina máquinas usando dados **sem rótulo**, tentando descobrir estrutura ou padrões.  
Não existem categorias pré-definidas nem respostas corretas fornecidas.

Objetivos:
- Encontrar padrões
- Agrupar itens parecidos
- Entender a estrutura escondida nos dados

Analogia:
- Você dá para uma criança uma tigela com bananas e laranjas e pede para separar.
- Mesmo sem saber as palavras "banana" e "laranja", ela consegue separar por tamanho, cor e forma.
- O aprendizado não supervisionado faz algo parecido.

Métodos comuns de aprendizado não supervisionado:
- Clusterização: agrupar pontos de dados parecidos, mesmo sem saber antes quais grupos existem
- Redução de dimensionalidade: reduzir o número de variáveis para simplificar dados complexos
- Modelagem de tópicos (topic modeling): descobrir temas em muitos textos
- Detecção de anomalias: achar comportamentos fora do padrão

Casos de uso:
- Segmentação de clientes no varejo
- Detecção de fraude no banco analisando padrões incomuns
- Recomendação de conteúdo em streaming
- Controle de qualidade na indústria (achar defeitos em linha de produção)

### Supervisionado vs. não supervisionado

Diferenças principais:

Aprendizado supervisionado:
- Faz previsões ou classificações com base em exemplos rotulados do passado  
- Precisa de dados rotulados  
- Risco: overfitting (o modelo "decorar" o treino e não generalizar)  

Aprendizado não supervisionado:
- Descobre padrões, estruturas e relações  
- Funciona com dados sem rótulo  
- Desafio: interpretar os resultados pode ser difícil e depende do algoritmo  

---

## 3. Engenharia de atributos (Feature Engineering)

Nesta parte, você vai:
- Entender o processo de engenharia de atributos
- Entender o impacto disso na performance do modelo

Engenharia de atributos é o processo de criar, transformar e selecionar os atributos (features) mais úteis para melhorar a performance do modelo.  
Isso vale tanto para aprendizado supervisionado quanto não supervisionado.

Ela usa conhecimento do domínio para transformar dados brutos em sinais que um modelo consegue entender.

Por que isso é importante:
- Melhora a precisão do modelo
- Torna as previsões mais interpretáveis
- Reduz ruído e complexidade

Exemplo:
- Você tem 30 atributos, mas talvez só "altura" realmente ajude a prever "peso".  
- Engenharia de atributos ajuda a descobrir quais atributos realmente importam.

Na área da saúde, combinar dados do paciente + exames + histórico pode gerar previsões mais precisas de risco de doenças.

### Técnicas principais

1. **Criação de atributos (feature creation)**  
   - Criar novos atributos usando conhecimento do negócio ou padrões observados  
   - Pode melhorar performance, reduzir impacto de outliers e aumentar interpretabilidade  

2. **Transformação de atributos (feature transformation)**  
   - Converter atributos em representações mais úteis  
   - Ajuda eficiência computacional  
   - Ajuda o modelo a capturar padrões mais profundos  
   - Exemplos: normalização, codificação de categorias, log/square root/reciprocal transform  

3. **Escalonamento de atributos (feature scaling)**  
   - Colocar atributos em escalas parecidas, para que um atributo com valores muito grandes não domine os outros  
   - Exemplos: min-max scaling, padronização (standard scaling), robust scaling  

4. **Seleção de atributos (feature selection)**  
   - Escolher só os atributos mais relevantes  
   - Reduz overfitting  
   - Melhora interpretação  
   - Reduz custo de processamento  
   - Métodos:
     - Métodos filtro (correlação estatística)
     - Métodos wrapper (testar subconjuntos)
     - Métodos embutidos (o próprio modelo escolhe, como regularização)

5. **Extração de atributos (feature extraction)**  
   - Combinar ou agregar atributos para gerar novos atributos mais informativos  
   - Normalmente reduz dimensionalidade  
   - Exemplos: redução de dimensionalidade, combinação de atributos, agregações, PCA (Análise de Componentes Principais)

### Ferramentas comuns

- Em R: dplyr, tidyr, caret  
- Em Python: pandas, NumPy, scikit-learn, featuretools  

---

## 4. Clusterização

Nesta parte, você vai:
- Entender o que é clusterização  
- Ver benefícios e desafios  

Clusterização é um exemplo de aprendizado não supervisionado.

Você trabalha com dados sem rótulo e tenta descobrir estrutura (grupos de pontos parecidos). Esses grupos são chamados de clusters.

Objetivo:
- Encontrar similaridade entre pontos de dados  
- Agrupar de forma significativa  
- Revelar segmentos naturais  

### Vantagens da clusterização

- Descobre padrões escondidos  
- Ajuda na visualização: mostra quais pontos "pertencem" juntos  
- Funciona mesmo quando rótulos seriam caros ou impossíveis de obter  
- Pode funcionar com dados de vários tipos (numéricos, categóricos), dependendo do algoritmo  

### Limitações e desafios

- Muitas vezes você precisa de etapas extras para validar se os grupos fazem sentido de verdade  
- Alguns algoritmos assumem formato ou tamanho específico dos clusters  
- Alguns não escalam bem quando há muitas variáveis (alta dimensionalidade)  
- Alguns não lidam bem com clusters de tamanhos muito diferentes  
- Misturar dados categóricos e numéricos pode ser problema dependendo do algoritmo  

### Para que usar clusterização?

Clusterização responde perguntas de similaridade, como:
- "Com qual grupo este cliente parece mais?"
- "Quais comportamentos aparecem juntos com frequência?"

Exemplos de uso:
- Agrupar e-mails/textos de clientes para medir satisfação  
- Monitorar a propagação de doenças  
- Encontrar padrões de compra no histórico de transações  
- Criar segmentos de marketing para campanhas direcionadas  

Importante:
- Clusterização pode gerar grupos que não são óbvios.
- Você ainda precisa interpretar e explicar o que cada grupo significa para o negócio.

### Exemplo do mundo real

Em 1997, um estudo sobre limite de crédito usou clusterização para separar clientes de cartão de crédito em cinco grupos.  
Entradas analisadas:
- Quanto pegavam emprestado  
- Frequência de atraso no pagamento  
- Hábitos de gasto  

Objetivo: prever o comportamento do cliente e ajustar marketing de forma mais eficiente.

---

## 5. Avaliação da acurácia de algoritmos de clusterização

Nesta parte, você vai:
- Ver como avaliar a qualidade de uma clusterização

Objetivo da clusterização:
- Maximizar a separação entre clusters (clusters bem separados)
- Minimizar a distância dentro de cada cluster (itens do mesmo cluster muito parecidos)

Conceitos úteis:
- Distância ou variância entre clusters: quão longe os clusters estão uns dos outros  
- Distância dentro do cluster: quão próximos estão os pontos dentro do mesmo grupo  

Uma forma de medir:
- Calcular a razão entre a variância entre clusters e a variância total  
- Quanto maior a separação, mais "forte" pode ser a clusterização  
- O cálculo exato depende do algoritmo

Outra ferramenta útil: gráfico "scree"
- Mostra quanta variância cada componente explica  
- Ajuda a enxergar quais dimensões (atributos) mais importam

### Perguntas que você deve fazer como gestor

- Como a distância foi medida? (por exemplo, distância Euclidiana, Manhattan etc.)  
- Os dados foram escalonados corretamente?  
  - Exemplo: se um atributo varia de 1 a 100 e outro de 0 a 1, isso pode distorcer o agrupamento sem normalização  
- Quantos clusters eram esperados?  
  - Se você queria 4 segmentos de marketing e recebeu 20 clusters, isso ainda é útil?  
- O algoritmo escala bem no volume de dados que você tem?  
  - Ele consegue rodar quase em tempo real conforme os dados crescem?  
- É possível interpretar os grupos resultantes?  
  - Se você não consegue explicar o que cada grupo significa, talvez o resultado não seja acionável

### Armadilhas e impactos

- Não existem rótulos, então não dá pra dizer "certo" ou "errado" como na classificação supervisionada  
- É preciso interpretar os clusters  
- Pode ser necessário usar métodos adicionais para confirmar se os padrões são reais

"Maldição da dimensionalidade":
- Quando você tem muitos atributos (muitas colunas):
  - Os dados ficam esparsos  
  - Distâncias deixam de ser tão significativas  
  - Fica mais difícil separar grupos de forma clara  

Alguns algoritmos:
- Sofrem com mistura de dados categóricos e numéricos  
- Assumem formatos específicos dos clusters  
- Não escalam bem quando o número de atributos cresce muito  

### Quando usar clusterização

Use clusterização quando:
1. Você tem dados sem rótulo  
2. Os dados têm vários atributos  
3. Você quer identificar padrões ou grupos naturais  
4. Você quer descobrir estruturas escondidas que não aparecem em gráficos simples  

---

## 6. Classificação

Nesta parte, você vai:
- Ver usos de classificação  
- Conhecer classificadores comuns  

Classificação é um tipo de aprendizado supervisionado.

Ela atribui novos exemplos (novos dados) a classes conhecidas com base em semelhança com dados rotulados usados no treino.

O modelo:
- Aprende uma relação entre entrada e saída usando dados rotulados  
- Usa essa relação para classificar novos dados

### Benefícios

- Ajuda a medir similaridade entre ideias, eventos, objetos ou pessoas  
- Organiza dados em rótulos claros  
- Suporta várias áreas (detecção de fraude, triagem médica etc.)  

### Desafios

- Pode sofrer overfitting: aprender ruído do treino e errar em dados novos  
- Precisa de dados rotulados, que são caros e demorados para produzir  
- O treino pode consumir bastante tempo e recurso computacional  

### Usos práticos

- Prever probabilidades (ex.: "70% de chance de ser spam" → então marcar como spam)  
- Prever pertencimento a um grupo  
- Dizer se um objeto ou pessoa é semelhante a outro padrão conhecido  

### Exemplo real

Um grande varejista analisou padrões de compra para prever quais clientes provavelmente estavam grávidos e então enviou cupons direcionados.  
Isso gerou preocupações éticas e de privacidade porque previu informações sensíveis de saúde sem comunicação explícita.

### Classificadores comuns

- **Regressão logística**  
  - Retorna probabilidades  
  - Fácil de interpretar  
  - Muito usada em finanças e saúde  

- **SVM (Support Vector Machine / Máquina de Vetor de Suporte)**  
  - Funciona bem quando as classes são separáveis  
  - Encontra a fronteira de decisão ótima  

- **Árvores de decisão**  
  - Tomam decisões seguindo regras do tipo "se ... então ..."  
  - Fáceis de explicar para áreas de negócio  
  - Base de modelos em conjunto como Random Forest e XGBoost  

- **k-vizinhos mais próximos (k-NN)**  
  - Baseado em distância  
  - Assume que itens parecidos ficam "perto" uns dos outros  
  - Classifica um ponto com base nas classes dos vizinhos mais próximos  

---

## 7. Avaliação da acurácia de modelos de classificação

Nesta parte, você vai:
- Ver fatores que afetam a acurácia de um modelo de classificação

Para avaliar um modelo de classificação:

1. Dividir os dados em:
   - Conjunto de treino (o modelo aprende aqui)
   - Conjunto de teste (o modelo é avaliado aqui)

2. Comparar previsões com resultados reais usando uma matriz de confusão.

### Termos da matriz de confusão

- Verdadeiro Positivo (VP / TP): previsto positivo e era positivo  
- Verdadeiro Negativo (VN / TN): previsto negativo e era negativo  
- Falso Positivo (FP): previsto positivo mas era negativo  
- Falso Negativo (FN): previsto negativo mas era positivo  

Também se analisa performance usando a curva ROC (Receiver Operating Characteristic):

- Mostra a taxa de verdadeiros positivos vs. taxa de falsos positivos em vários limiares de probabilidade  
- Cada limiar gera uma matriz de confusão diferente  

AUC (Área sob a Curva ROC):
- Mede a capacidade geral do modelo de separar classes  
- AUC maior que 0,5 significa "melhor que chute aleatório"  

### Perguntas que você deve fazer como gestor

- Como foi escolhida a métrica de distância (se aplicável)?  
- Como os dados foram divididos entre treino e teste?  
- Os dados foram normalizados/escalonados?  
- Quais limiares (thresholds) foram usados para calcular ROC e AUC?  

### Quando usar classificação

Use classificação quando:
- Você tem dados rotulados  
- Você quer prever a qual grupo algo pertence  
- Você quer prever comportamentos ou eventos futuros  
- Você quer entender quais variáveis estão influenciando a decisão  

---

## 8. Regressão

Nesta parte, você vai:
- Entender o que é regressão  
- Ver benefícios e desafios  

Regressão cria uma linha de melhor ajuste para um conjunto de dados e prevê o valor de uma variável com base em outra.

Ela analisa relações entre variáveis.

Diferença para classificação:
- Regressão prevê valores numéricos contínuos  
- Classificação prevê categorias (classes)

### Benefícios e desafios

Benefícios:
- Pode fornecer uma forma objetiva de prever eventos  
- Ajuda a priorizar quais fatores mais influenciam o resultado  
- Ajuda a guiar quais dados devem ser coletados  

Desafios:
- Dados faltantes podem atrasar análise  
- Construir e manter modelos pode ser caro  
- Exige conhecimento do domínio para interpretar corretamente  
- Alta variabilidade ou problema mal definido pode gerar incerteza sobre qual abordagem de modelagem usar  

### Perguntas que a regressão pode responder

- Quais fatores importam mais?  
- Quais fatores podem ser ignorados?  
- Existe interação entre fatores?  
- Quão confiantes estamos nessas conclusões?  

---

## 9. Regressão linear simples

Nesta parte, você vai:
- Ver casos de uso de regressão linear simples

Tópicos:
- Regressão linear simples  
- Medição de erro  
- Tratamento de outliers  
- Medindo acurácia  
- Métricas comuns de avaliação em regressão  

Regressão linear simples é muitas vezes o primeiro modelo que cientistas de dados aprendem.

Objetivo:
- Encontrar a linha que melhor se ajusta aos dados  
- Minimizar a distância entre essa linha e cada observação  
- Usar essa linha para prever resultados com base em um único preditor  

Passos básicos:
1. Coletar os dados  
2. Plotar os dados para ver se há um padrão aproximadamente linear  
3. Ajustar a linha de melhor ajuste  

### Medindo erro

Jeitos de medir erro:
- Variância: quão distantes os valores reais estão dos valores esperados  
- Aleatoriedade: os erros são aleatórios ou existe viés?  
- Desvio padrão / certeza: quão concentrados os valores estão em torno da previsão esperada  

### Outliers

Outliers (valores muito fora do padrão) podem distorcer muito a inclinação (slope) e o intercepto da linha de regressão.

Ferramentas para detectar outliers:
- Gráfico de dispersão (scatterplot)  
- Boxplot (box-and-whisker)  
- Distância de Cook (Cook’s distance)  

### Métricas de acurácia

- Covariância: como duas variáveis variam juntas  
- Correlação: versão normalizada da covariância, vai de -1 a +1  
- Valor-p (p-value): quão provável seria observar essa relação se, na verdade, não existisse relação linear na população  
- R-quadrado (R²): porcentagem da variação nos dados explicada pelo modelo  

Duas métricas comuns de erro:

- RMSE (Root Mean Squared Error / Raiz do Erro Quadrático Médio)  
  - Usa os resíduos (erros) ao quadrado  
  - Quanto menor, melhor  

- MAE (Mean Absolute Error / Erro Médio Absoluto)  
  - Média do valor absoluto da diferença entre previsto e real  
  - Aumenta conforme os erros aumentam  

---

## 10. Regressão linear múltipla

Nesta parte, você vai:
- Entender como usar regressão linear múltipla

Regressão linear múltipla estende a regressão linear simples para incluir mais de uma variável independente (mais de um preditor).

Objetivo:
- Entender como vários preditores, juntos, afetam a variável alvo

### Problemas adicionais em regressão múltipla

- Multicolinearidade:  
  - Dois ou mais preditores são altamente correlacionados entre si  
  - Você pode "contar duas vezes" o mesmo efeito sem perceber  

- Autocorrelação:  
  - Os erros (resíduos) não são independentes  
  - A observação atual está relacionada com observações anteriores  

- Heterocedasticidade:  
  - A variância dos erros não é constante  
  - Ou seja, o quanto o modelo erra depende do nível do preditor  

### Perguntas que você deve fazer como gestor

- Nós entendemos a distribuição dos dados?  
- Outliers foram identificados? Eles eram importantes? Foram removidos?  
- As variáveis foram testadas para multicolinearidade (para não contar o mesmo efeito duas vezes)?  
- Qual foi o R-quadrado (R²)?  

---

## 11. Desafios em Machine Learning

Nesta parte, você vai:
- Ver desafios comuns em ML

Desafios comuns:

1. Poucos dados de treino  
   - Leva a overfitting  
   - O modelo aprende ruído em vez de padrão real  
   - Alta variância: vai muito bem no treino e mal em dados novos  
   - Possíveis soluções:
     - Coletar mais dados  
     - Gerar dados sintéticos (por exemplo, técnicas de oversampling)  

2. Dados de treino não representativos  
   - Você treina com um subconjunto limitado da população  
   - Se esse subconjunto é pequeno ou enviesado, o modelo não generaliza  
   - Isso cria viés de amostragem  
   - O objetivo é equilibrar viés e variância (trade-off viés/variância)  

3. Qualidade ruim dos dados  
   - Valores ausentes  
   - Erros  
   - Ruído  
   - Duplicados  
   - Registros incompletos  
   - Soluções:
     - Limpeza de dados  
     - Remoção ou tratamento de outliers  
     - Deduplicação  
     - Tratamento de valores ausentes  

4. Atributos irrelevantes  
   - "Lixo entra, lixo sai"  
   - Se você fornece atributos irrelevantes, você recebe previsões ruins  
   - Soluções:
     - Engenharia de atributos  
     - Seleção de atributos:
       - Manter só os atributos relevantes  
       - Técnicas:
         - Regularização (Ridge, Lasso)  
         - Importância de atributos em Random Forests  
         - Seleção estatística  
     - Extração de atributos:
       - Criar novos atributos a partir de conhecimento do domínio  
       - Redução de dimensionalidade (por exemplo, PCA)  
       - Combinação / agregação de atributos para gerar sinais mais fortes  

---

## 12. Resumo do curso

Neste conteúdo:
- Você definiu o que é machine learning e onde ele é usado  
- Você viu métodos fundamentais de ML:
  - Clusterização  
  - Classificação  
  - Regressão  
- Você aprendeu como avaliar a performance de modelos  
- Você revisou desafios comuns de ML e estratégias para lidar com eles  

---

Copyright 2025 Skillsoft Ireland Limited. Todos os direitos reservados.



https://cdn2.percipio.com/secure/c/1761497995.80b76f0f861e0e1d28122c32a99255eff6b17ce9/eot/transcripts/a583c2bd-30b5-4797-be49-c85586632644/it_aidfamdj_01_enus.html
