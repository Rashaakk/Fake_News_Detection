# Fake_News_Detection
📰 Fake News Detection Using NLP Classifiers

Identify fake vs. real news articles using the LIAR dataset. Leverages linguistic features, TF-IDF, and transformer-based BERT embeddings with Logistic Regression, Gradient Boosting, and DistilBERT models.

----------------------------------------------------------------------------------------------------

👥 Team Members

Athulya Mannambath (Msc Computer Science in Data Analytics)

Amrutha (Mcs Data Analytics & Computational Science)

Rasha KK  (Msc Data Analytics & Computational Science)

----------------------------------------------------------------------------------------------------

📌 **Problem Statement & Motivation**

Misinformation and fake news have become a critical challenge in the digital age, influencing public opinion, elections, and societal trust. Manual fact-checking is slow and does not scale. This project builds an automated fake news detection system that classifies political statements as real or fake using Natural Language Processing techniques.
We explore whether linguistic patterns alone — such as readability, punctuation use, and word choice — can distinguish fake news from real news, and compare classical ML models against modern transformer-based architectures.

--------------------------------------------------------------------------------------------------

📊 Dataset
Source: 

LIAR Dataset — a benchmark dataset for fake news detection.

SplitSize:

Train-->10,239 statements

Validation-->1,283 statements

Test-->1,266 statements


Features:

statement — the political claim being fact-checked

label — original 6-class label: pants-fire, false, barely-true, half-true, mostly-true, true

speaker, subject, party_affiliation, context — metadata

Class Distribution (Binary):
ClassLabels MappedFake (0)pants-fire, false, barely-trueReal (1)half-true, mostly-true, true

----------------------------------------------------------------------------------------------------

**🔬 Methodology**
1. Data Preprocessing

Lowercasing, URL removal, digit stripping, punctuation removal
Contraction expansion (e.g., don't → do not)
Stopword removal using NLTK

2. Linguistic Feature Engineering
Eight hand-crafted features extracted per statement:
FeatureDescriptionnum_charsTotal character countnum_wordsWord countnum_sentencesSentence countavg_word_lengthMean word lengthexclamation_countNumber of !question_countNumber of ?capital_ratioRatio of uppercase charactersreadabilityFlesch Reading Ease score

3. Exploratory Data Analysis

Label distribution visualisation
Per-feature box plots comparing Fake vs. Real news
Word clouds for fake and real news corpora

**Classical Machine Learning Models**

* Combined feature matrix constructed by stacking word-level TF-IDF (unigrams and bigrams), character-level TF-IDF (2–4 grams), and the eight hand-crafted linguistic features
* Logistic Regression trained on the combined matrix with class balancing to handle label skew
* Hyperparameters tuned using GridSearchCV with 5-fold stratified cross-validation
* XGBoost used as the gradient boosting classifier, selected for its efficiency on sparse matrices
* XGBoost optimised using RandomizedSearchCV across learning rate, tree depth, and subsampling ratio
* Both models evaluated using accuracy, weighted F1-score, and ROC-AUC on the held-out test set
* Feature importance analysis performed using logistic regression coefficients to identify top fake and real news indicators
* SHAP values used to interpret XGBoost predictions at the feature level
* Cross-domain generalisation tested by evaluating on health and science topic statements after training on political statements






**Transformer Models and Deep Learning
Overview**

This section implements transformer-based fake news detection using BERT/DistilBERT models. The transformer module improves contextual understanding of news articles compared to traditional machine learning models.

**Implemented Components**
* BERT embedding extraction
* DistilBERT transformer classifier
* Fine-tuning transformer models
* Attention visualization
* Comparative evaluation with classical ML models
* Cross-domain generalization analysis

**Technologies Used**
* PyTorch
* Hugging Face Transformers
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn

**Transformer Models Used**
* DistilBERT
* BERT Embedding + Logistic Regression

**Workflow**
* Load preprocessed dataset from Member 1 outputs
* Generate contextual embeddings using DistilBERT
* Train transformer-based classifier
* Evaluate performance on validation and test datasets
* Compare results with classical machine learning models
* Visualize training curves, confusion matrix, and attention scores

**Evaluation Metrics**

The following metrics are used:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Outputs Generated

**The transformer pipeline generates:**

* Model evaluation reports
* Confusion matrix
* Training loss curves
* Radar chart comparison
* Attention visualization
* Final comparative performance plots


**Observations**

Classical ML Models


| Model | AccuracyPrecision (Fake) | Recall (Fake) | F1 (Fake) | Precision (Real) | Recall (Real) | F1 (Real) | Weighted F1 |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|
| Logistic Regression | 0.60 | 0.54 | 0.57 | 0.55 | 0.65 | 0.63 | 0.64 | 0.61 |
| Gradient Boosting | 0.59 | 0.59 | 0.21 | 0.31 | 0.59 | 0.89 | 0.71 | 0.53 |

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | 0.60 | 0.61 |
| Gradient Boosting | 0.59 | 0.53 |

* Logistic Regression produced the most balanced results across both classes making it reliable for detecting both.
* Gradient Boosting is heavily biased toward the Real class meaning it misses most fake news articles.
* DistilBERT outperformed all classical models with the highest accuracy and F1, demonstrating the advantage of contextual embeddings over bag-of-words features.
* BERT Embeddings + LR offered a strong middle ground — better than classical ML without the overhead of full fine-tuning.


**Streamlit Deployment**
The fake news detection application was also deployed using Streamlit to provide an interactive user interface for real-time news classification.
- https://fakenewsdetection-dm4pqnqpxdxhoizqrozpic.streamlit.app/

**Conclusion**

This project successfully implemented a fake news detection system using both classical machine learning and transformer-based deep learning models. Traditional NLP techniques such as TF-IDF, readability analysis, and linguistic feature extraction were combined with advanced transformer architectures like DistilBERT to improve classification performance.

The transformer-based models demonstrated better contextual understanding of news articles and achieved strong performance in detecting fake and real news. Comparative evaluation showed that BERT-based approaches can outperform traditional models in handling complex linguistic patterns and semantic relationships.

The project also explored:

* Attention visualization for interpretability
* Cross-domain generalization capability
* Comparative analysis between classical ML and transformer models

Overall, the project highlights the effectiveness of NLP and transformer models in combating misinformation and fake news detection, while also providing insights into linguistic patterns commonly associated with fake news articles.
