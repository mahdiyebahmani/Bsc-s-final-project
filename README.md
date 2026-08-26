Persian Sentiment Analysis of Digikala Reviews

This repository contains a bachelor's final project focused on sentiment classification of Persian Digikala product reviews.

The project implements an end-to-end NLP pipeline including Persian text preprocessing, feature extraction, model training, and evaluation.

Main approaches:

* BOW and TF-IDF with Unigram and Bigram features
* FastText word embeddings
* Classical models: Multinomial Naive Bayes, Logistic Regression, and Linear SVM
* Neural models: MLP, CNN, and Bidirectional GRU
* Transformer-based classification using ParsBERT

The dataset is classified into three classes:

* `not_recommended`
* `no_idea`
* `recommended`

Evaluation is performed using Accuracy, Precision, Recall, and Macro F1, with Macro F1 used as the primary metric due to class imbalance.

Technologies: Python, Scikit-learn, Hazm, FastText, TensorFlow/Keras, PyTorch, and Hugging Face Transformers.

The main objective is to compare different preprocessing methods, text representations, and learning architectures for Persian sentiment classification and identify the most effective approach for the dataset.
