# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:01:25 2020

@author: Renan
"""

'''Importar Dataset'''

from sklearn.datasets import fetch_20newsgroups
import numpy as np

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
twenty_test = fetch_20newsgroups (subset='test', shuffle=True)

'''Pre-Processing - Pipeline'''
#Contagem
#Frequencia
#Classificador SVM

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2),stop_words='english')),
                     ('tfidf', TfidfTransformer(use_idf=True)),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=0.001, max_iter=5, random_state=42)),
])

'''Treino'''

text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

'''Teste'''

predicted_svm = text_clf.predict(twenty_test.data)
accuracy_SVM = np.mean(predicted_svm == twenty_test.target)

'''Avaliacao'''

from sklearn.metrics import confusion_matrix, accuracy_score

precisao = accuracy_score(twenty_test.target, predicted_svm)
matriz = confusion_matrix(twenty_test.target, predicted_svm)




