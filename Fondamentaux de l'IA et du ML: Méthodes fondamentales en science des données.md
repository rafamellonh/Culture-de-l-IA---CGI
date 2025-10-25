# Fundamentals of AI & ML: Foundational Data Science Methods

Data science methods are used across several industries to deliver value to businesses. Machine learning (ML) is a data science method that uses prediction algorithms that find patterns in massive amounts of data, allowing machines to predict future results and make decisions with minimal human intervention.

This document covers foundational methods for using machine learning.

You will examine what machine learning is, how it is categorized, and some everyday use cases for supervised and unsupervised machine learning. Then you will see feature engineering and its impact on model performance. Next, you will focus on common types of machine learning tasks, such as clustering, classification, and simple and multiple linear regression. Finally, you will explore various machine learning challenges and how to overcome them.

Upon completion, you will be able to define machine learning and methods for using it.

---

## Table of Contents

1. Course Overview  
2. Machine Learning (ML)  
3. Feature Engineering  
4. Clustering  
5. Evaluation of Clustering Algorithm Accuracy  
6. Classification  
7. Evaluation of Classification Model Accuracy  
8. Regression  
9. Simple Linear Regression  
10. Multiple Linear Regression  
11. Machine Learning Challenges  
12. Course Summary  

---

## 1. Course Overview

In this section:
- Common data science methods
- Use cases for these methods
- How to evaluate model performance
- What machine learning (ML) is
- Core ML tasks: clustering, classification, regression, and feature engineering

Data science methods are used across several industries to deliver value to businesses. This course walks through those methods and how to evaluate them.

---

## 2. Machine Learning (ML)

In this section, you will:
- Identify use cases for machine learning
- Differentiate supervised and unsupervised machine learning

Machine learning is a type of artificial intelligence (AI) that uses algorithms to find patterns in massive amounts of data and predict future results with minimal human intervention.

Machine learning algorithms use historical data to make predictions. Most machine learning is categorized as supervised or unsupervised.

### Real-world uses of ML

Machine learning powers many services:

- Recommendation systems (streaming, shopping)
- Product recommendations in e-commerce
- Search engines
- Social media feeds (ranking content)
- Voice assistants

Other areas:
- Transportation and logistics: predictive maintenance
- Healthcare: identifying patient subgroups for targeted care using techniques such as natural language processing
- Finance: detecting fraudulent transactions using clustering
- Sales and marketing: predicting churn rate and customer lifetime value

### Supervised Learning

Supervised learning trains machines using labeled data so they can make predictions on new or unknown data.

- You have input variables (X) and an output variable (Y)
- You train an algorithm to learn the mapping function between X and Y
- The goal is: given new X, predict Y

Labeled data means the data points are tagged with the correct answers (for example, "spam" or "not spam").

Examples of supervised learning tasks:
- Classification (image classification, spam detection, fraud detection)
- Regression (forecasting, weather prediction, GDP prediction, stock market prediction)
- Natural language processing (speech recognition, sentiment analysis, chatbots, translation)
- Recommendation systems
- Deep learning (text generation, object detection, facial recognition)

Example:
- Emails classified as spam or not spam based on how their features compare to emails that a human marked as spam.

Other examples:
- Medical diagnosis from images
- Automatic image tagging
- Object and pedestrian recognition for self-driving cars

### Unsupervised Learning

Unsupervised learning trains machines using unlabeled data to discover structure or patterns. There are no predefined labels or target outcomes.

The goal is to:
- Find patterns
- Group similar things
- Understand the structure of the data

Analogy:
- Give a child a bowl with bananas and oranges and ask them to separate them.
- The child can group them by size, color, and shape, even without knowing the words "banana" or "orange".
- Unsupervised learning does something similar.

Common unsupervised methods:
- Clustering: group similar data points, even if you don’t know in advance what those groups should be
- Dimensionality reduction: reduce the number of features to simplify complex data
- Topic modeling: discover topics in large sets of text documents
- Anomaly detection: find unusual events or behavior

Example use cases:
- Customer segmentation in retail
- Fraud detection in banking by identifying unusual spending patterns
- Content recommendation in streaming platforms
- Quality control in manufacturing (detecting defects on a production line)

### Supervised vs. Unsupervised

Key differences:

Supervised ML:
- Predicts or classifies based on past labeled examples  
- Requires labeled data  
- Risk: overfitting if there's not enough data  

Unsupervised ML:
- Discovers patterns, structures, or relationships  
- Works with unlabeled data  
- Challenge: interpretability can be hard and depends on algorithm choice  

---

## 3. Feature Engineering

In this section, you will:
- Outline the process of feature engineering
- Understand its impact on model performance

Feature engineering is the process of creating, transforming, and selecting the most useful features (inputs) to improve model performance. It applies to both supervised and unsupervised learning.

It uses domain knowledge to turn raw data into signals that a model can understand.

Why it matters:
- Improves model accuracy
- Makes predictions more interpretable
- Reduces noise and complexity

Example:
- You have 30 features, but maybe only "height" is truly predictive of "weight".
- Feature engineering helps identify which features matter.

In healthcare, combining patient details, test results, and history can produce more accurate disease predictions.

### Main techniques

1. Feature creation  
   - Create new features using domain knowledge or observed patterns  
   - Can improve performance, robustness to outliers, and interpretability  

2. Feature transformation  
   - Convert features into a more useful representation  
   - Improves computational efficiency  
   - Helps the model capture deeper patterns  
   - Examples: normalization, encoding categorical values, log/square root/reciprocal transforms  

3. Feature scaling  
   - Standardize ranges so that large-scale features don't dominate the model  
   - Examples: min-max scaling, standard scaling, robust scaling  

4. Feature selection  
   - Choose only the most relevant features  
   - Reduces overfitting  
   - Improves interpretability  
   - Reduces compute cost  
   - Methods:
     - Filter methods (statistical relationships)
     - Wrapper methods (test subsets)
     - Embedded methods (learned during training, e.g. regularization)

5. Feature extraction  
   - Combine or aggregate features to produce new, more informative features  
   - Often reduces dimensionality  
   - Examples: dimensionality reduction, feature combination, aggregation, PCA (Principal Component Analysis)

### Tools

- R: dplyr, tidyr, caret  
- Python: pandas, NumPy, scikit-learn, featuretools  

---

## 4. Clustering

In this section, you will:
- Outline clustering
- Understand its benefits and challenges

Clustering is an example of unsupervised machine learning.

You work with unlabeled data and try to discover structure (groups of similar data points). These groups are called clusters.

The goal is to:
- Find similarities between data points
- Group them meaningfully
- Reveal natural segments

### Pros of clustering

- Can discover hidden patterns  
- Helps with visualization by highlighting which points belong together  
- Works when labels are expensive or impossible to obtain  
- Can work with different types of data (categorical, numeric), depending on algorithm  

### Challenges and limitations

- You may need other methods to validate that the discovered patterns are real and meaningful  
- Some algorithms make assumptions about cluster shape and size  
- Some algorithms don’t scale well to high-dimensional data (many variables)  
- Some methods don’t handle clusters of very different sizes well  
- Mixing categorical and numeric data may be an issue for certain algorithms  

### What can clustering be used for?

Clustering answers similarity questions like:
- "Who is this customer most similar to?"
- "Which behaviors tend to group together?"

Example uses:
- Grouping raw email text to identify customer satisfaction signals
- Tracking the spread of disease
- Finding shopping patterns in purchase history
- Marketing segmentation (for targeted campaigns)

Domain expertise is critical:
- Clustering may reveal unintuitive groups
- You still need to interpret what those patterns mean

### Real-world example

A 1997 study on credit line optimization used clustering to group credit card customers into five categories.  
Inputs included:
- How much customers borrowed
- How often they were late
- How often they spent

The goal was to predict customer behavior and tailor marketing more effectively.

---

## 5. Evaluation of Clustering Algorithm Accuracy

In this section, you will:
- List considerations for evaluating the accuracy of a clustering algorithm

Goal of clustering:
- Maximize separation between clusters (clusters are far apart)
- Minimize distance within clusters (points in the same cluster are similar)

Useful concepts:
- Inter-cluster distance or variance: distance between clusters  
- Intra-cluster distance: distance within a cluster  

One way to assess performance:
- Compute the ratio of inter-cluster variance to total variance  
- Higher separation can indicate more meaningful clusters  
- Exact details depend on the algorithm  

A scree plot can also help:
- Shows how much variance each component explains  
- Helps identify which features or dimensions are most important  

### Questions to ask as a manager

- How was distance measured? (for example, Euclidean distance or Manhattan distance)
- Was the data scaled properly?  
  - Example: if feature A ranges from 1–100 and feature B from 0–1, unscaled data can distort results
- How many clusters were expected?  
  - If you only want 4 audience segments but got 20 clusters, is that useful?
- Does it scale to your data volume and speed needs?  
  - Can it run in near real-time as data grows?
- Can you interpret the resulting groups?  
  - If you can’t explain what the clusters mean, the result may not be useful

### Implications and pitfalls

- There are no labels, so you can’t say “correct vs incorrect” as in supervised learning  
- Interpretation is required to understand the meaning of clusters  
- You may need more methods to confirm the patterns you find  

Curse of dimensionality:
- As the number of features grows:
  - Data becomes sparse
  - Distances become less meaningful
  - It can get harder to identify distinct clusters

Some algorithms:
- Struggle with a mix of categorical and numeric data
- Assume certain shapes of clusters
- Do not scale well to lots of features

### When to use clustering

Use clustering when:
1. You have unlabeled data  
2. The data has multiple attributes  
3. You want to identify patterns or natural groupings  
4. You want to detect hidden structures not obvious from simple charts  

---

## 6. Classification

In this section, you will:
- Identify uses of classification
- Name common classifiers

Classification is a type of supervised machine learning.

It assigns new data points to known classes based on similarity to labeled training data.

The model:
- Learns a mapping from inputs to outputs using training data
- Applies that learned mapping to classify new data

### Benefits

- Can determine similarity across ideas, events, objects, or people  
- Organizes data into clear labels  
- Supports many domains (fraud detection, medical triage, etc.)  

### Challenges

- Can overfit to noise in the training data and then fail on new data  
- Needs labeled data, which can be expensive or time-consuming to produce  
- Training can take significant time, depending on the algorithm  

### Uses

- Predicting probabilities (for example, “70% chance this is spam” which is then used to classify the email as spam)
- Group membership prediction
- Detecting whether an object or person is similar to another

### Real-world example

A large retailer analyzed customer purchase patterns to predict which customers were likely pregnant and then sent targeted coupons. This raised ethical and privacy concerns because sensitive health-related predictions were inferred without explicit disclosure.

### Common classifiers

- Logistic regression  
  - Outputs probabilities  
  - Highly interpretable  
  - Common in finance and healthcare  

- Support Vector Machines (SVM)  
  - Works well when classes are clearly separable  
  - Finds the optimal decision boundary (hyperplane)  

- Decision trees  
  - Follow a tree-like set of rules to assign classes  
  - Easy to interpret and explain  
  - Foundation of ensemble methods like Random Forests and XGBoost  

- k-nearest neighbors (k-NN)  
  - Distance-based  
  - Assumes similar things exist in close proximity  
  - Classifies a point based on the classes of its neighbors  

---

## 7. Evaluation of Classification Model Accuracy

In this section, you will:
- Consider factors that affect the accuracy of a classification model

To evaluate a classification model:

1. Split data into:
   - Training set (model learns here)
   - Test set (model is evaluated here)

2. Compare predictions vs. actual outcomes using a confusion matrix.

### Confusion Matrix Terms

- True Positive (TP): predicted positive, actually positive  
- True Negative (TN): predicted negative, actually negative  
- False Positive (FP): predicted positive, actually negative  
- False Negative (FN): predicted negative, actually positive  

You can also analyze performance using the ROC curve (Receiver Operator Characteristic):

- Plots true positive rate vs. false positive rate at different probability thresholds  
- Each threshold produces a different confusion matrix  

AUC (Area Under the ROC Curve):
- Measures overall ability to separate classes  
- AUC greater than 0.5 means better than random guessing  

### Questions to ask as a manager

- How was the distance measure identified?  
- How was data split into training and test sets?  
- Was the data scaled?  
- What thresholds were used for ROC and AUC scoring?  

### When to use classification

Use classification when:
- You have labeled data  
- You want to predict group assignments  
- You want to predict behaviors or events  
- You want insight into which features drive predictions  

---

## 8. Regression

In this section, you will:
- Outline regression
- Understand its benefits and challenges

Regression creates a line of best fit for a dataset and predicts the value of one variable based on another.

It examines relationships between variables.

Difference from classification:
- Regression predicts numeric values
- Classification predicts categories

### Benefits and challenges

Benefits:
- Can provide an objective way to predict events  
- Helps prioritize which factors matter most  
- Helps guide data collection  

Challenges:
- Missing data can slow analysis  
- Building and maintaining models can be costly  
- Requires industry and domain knowledge to interpret correctly  
- High variability or unclear problem definition can create uncertainty in the correct modeling approach  

### Questions regression can answer

- Which factors matter the most?  
- Which factors can be ignored?  
- Do factors interact with each other?  
- How confident are we in these conclusions?  

---

## 9. Simple Linear Regression

In this section, you will:
- Name use cases for simple linear regression

Topics:
- Simple linear regression  
- Measuring error  
- Handling outliers  
- Determining accuracy  
- Common regression evaluation metrics  

Simple linear regression is often the first model data scientists learn.

Goal:
- Find a line that best fits the data  
- Minimize the distance between that line and each observation  
- Use that line to predict outcomes based on a single predictor  

Basic steps:
1. Gather data  
2. Plot data to check for linear patterns  
3. Draw the best-fit line  

### Measuring error

Ways to measure error:
- Variance: how widely actual values deviate from expected values  
- Randomness: are errors random, or is there bias?  
- Standard deviation and certainty: how tightly values fall within expected ranges  

### Outliers

Outliers can distort the regression line (slope and intercept).

Tools to detect outliers:
- Scatterplots  
- Box-and-whisker plots  
- Cook’s distance  

### Accuracy metrics

- Covariance: how two variables move together  
- Correlation: scaled version of covariance, ranges from -1 to +1  
- P-value: how likely it is to observe the result you saw if there is actually no true linear relationship in the population  
- R-squared: percentage of the variance in the data explained by the model  

Two common error metrics:
- RMSE (Root Mean Squared Error)  
  - Uses residuals (errors)  
  - Lower is better  

- MAE (Mean Absolute Error)  
  - Average absolute difference between predicted and actual  
  - Increases as errors increase  

---

## 10. Multiple Linear Regression

In this section, you will:
- Outline how to use multiple linear regression

Multiple linear regression extends linear regression to include more than one independent variable (more than one predictor).

Goal:
- Understand how several predictors together affect a target variable

### Added concerns in multiple regression

- Multicollinearity:
  - Two or more predictors are strongly correlated with each other  
  - You might “double count” the same effect  

- Autocorrelation:
  - Errors (residuals) are not independent  
  - The current observation is related to past observations  

- Heteroskedasticity:
  - The spread (variance) of the errors is not consistent across the range of values  
  - In other words, error variance changes depending on the level of the predictor  

### Questions to ask as a manager

- Do we understand the underlying data distribution?  
- Did we identify outliers? Were they significant? Were they removed?  
- Were variables tested for multicollinearity, so we're not double-counting?  
- What was the R-squared?  

---

## 11. Machine Learning Challenges

In this section, you will:
- List common machine learning challenges

Common challenges:

1. Insufficient training data  
   - Leads to overfitting  
   - The model learns noise instead of general patterns  
   - High variance: performs well on training data, badly on new data  
   - Possible solutions:
     - Collect more data  
     - Generate synthetic samples (for example, oversampling methods)  

2. Non-representative training data  
   - You train on a subset of the population  
   - If the subset is too small or biased, it will not generalize  
   - This creates sampling bias  
   - Goal is to balance bias and variance (bias-variance tradeoff)  

3. Poor data quality  
   - Missing values  
   - Errors  
   - Noise  
   - Duplicates  
   - Incomplete records  
   - Solutions:
     - Clean data  
     - Remove or bound outliers  
     - Deduplicate  
     - Handle missing values  

4. Irrelevant features  
   - "Garbage in, garbage out"  
   - If you feed irrelevant features, you get irrelevant predictions  
   - Solutions:
     - Feature engineering  
     - Feature selection:
       - Keep only relevant features  
       - Techniques:
         - Regularization (Ridge, Lasso)  
         - Random forest feature importance  
         - Statistical feature selection  
     - Feature extraction:
       - Create new features using domain knowledge  
       - Use dimensionality reduction (for example, PCA)  
       - Combine or aggregate features to produce more informative signals  

---

## 12. Course Summary

In this course:
- You defined machine learning and explored where it is used  
- You learned foundational ML methods:
  - Clustering  
  - Classification  
  - Regression  
- You learned how to evaluate model performance  
- You reviewed common ML challenges and strategies to address them  

---

Copyright 2025 Skillsoft Ireland Limited. All rights reserved.
