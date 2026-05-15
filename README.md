# Fake_News_Detection
📰 Fake News Detection Using NLP Classifiers

Identify fake vs. real news articles using the LIAR dataset. Leverages linguistic features, TF-IDF, and transformer-based BERT embeddings with Logistic Regression, Gradient Boosting, and DistilBERT models.

----------------------------------------------------------------------------------------------------

👥 Team Members
-----------------------------------------------------------------
|Athulya Mannambath | Msc Computer Science in Data Analytics    |
-----------------------------------------------------------------
|Amrutha            | Mcs Data Analytics & Computational Science| 

----------------------------------------------------------------|
|Rasha              | Mcs Data Analytics & Computational Science| 
-----------------------------------------------------------------

----------------------------------------------------------------------------------------------------

📌 Problem Statement & Motivation

Misinformation and fake news have become a critical challenge in the digital age, influencing public opinion, elections, and societal trust. Manual fact-checking is slow and does not scale. This project builds an automated fake news detection system that classifies political statements as real or fake using Natural Language Processing techniques.
We explore whether linguistic patterns alone — such as readability, punctuation use, and word choice — can distinguish fake news from real news, and compare classical ML models against modern transformer-based architectures.

----------------------------------------------------------------------------------------------------



















Transformer Models and Deep Learning
Overview

This section implements transformer-based fake news detection using BERT/DistilBERT models. The transformer module improves contextual understanding of news articles compared to traditional machine learning models.

Implemented Components
BERT embedding extraction
DistilBERT transformer classifier
Fine-tuning transformer models
Attention visualization
Comparative evaluation with classical ML models
Cross-domain generalization analysis

Technologies Used
PyTorch
Hugging Face Transformers
Scikit-learn
Pandas
NumPy
Matplotlib
Seaborn

Transformer Models Used
DistilBERT
BERT Embedding + Logistic Regression

Workflow
Load preprocessed dataset from Member 1 outputs
Generate contextual embeddings using DistilBERT
Train transformer-based classifier
Evaluate performance on validation and test datasets
Compare results with classical machine learning models
Visualize training curves, confusion matrix, and attention scores

Evaluation Metrics

The following metrics are used:

Accuracy
Precision
Recall
F1-Score
ROC-AUC
Outputs Generated

The transformer pipeline generates:

Model evaluation reports
Confusion matrix
Training loss curves
Radar chart comparison
Attention visualization
Final comparative performance plots

Streamlit Deployment
The fake news detection application was also deployed using Streamlit to provide an interactive user interface for real-time news classification.
https://fakenewsdetection-dm4pqnqpxdxhoizqrozpic.streamlit.app/

Conclusion

This project successfully implemented a fake news detection system using both classical machine learning and transformer-based deep learning models. Traditional NLP techniques such as TF-IDF, readability analysis, and linguistic feature extraction were combined with advanced transformer architectures like DistilBERT to improve classification performance.

The transformer-based models demonstrated better contextual understanding of news articles and achieved strong performance in detecting fake and real news. Comparative evaluation showed that BERT-based approaches can outperform traditional models in handling complex linguistic patterns and semantic relationships.

The project also explored:

Attention visualization for interpretability
Cross-domain generalization capability
Comparative analysis between classical ML and transformer models

Overall, the project highlights the effectiveness of NLP and transformer models in combating misinformation and fake news detection, while also providing insights into linguistic patterns commonly associated with fake news articles.
