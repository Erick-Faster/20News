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

'''V1 - Bag of words model não compacto'''

'''Contagem'''
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words='english', ngram_range=(1,2))
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape

'''Frequencia'''
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer(use_idf=True)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

from sklearn import svm
clf = svm.SVC(kernel= 'linear', gamma = 0.01, C = 0.001 )

'''Treino'''

text_clf = clf.fit(X_train_tfidf, twenty_train.target)

'''Teste'''

predicted_svm = text_clf.predict(twenty_test.data)
accuracy_SVM = np.mean(predicted_svm == twenty_test.target)

'''Avaliacao'''

from sklearn.metrics import confusion_matrix, accuracy_score

precisao = accuracy_score(twenty_test.target, predicted_svm)
matriz = confusion_matrix(twenty_test.target, predicted_svm)
