# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:26:26 2020

@author: Renan
"""

from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

'''V1 - Bag of words model não compacto'''

'''Contagem'''
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape

'''Frequencia'''
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

'''ML'''

from keras.models import Sequential
from keras.layers import Dense

text_clf = Sequential() #Camadas sao adicionadas sequencialmente

#Cria ligacoes densas na camada de ENTRADA
text_clf.add(Dense(units = 128, #No de entradas (30) + no saidas (1) / 2
                        activation = 'relu',
                        kernel_initializer = 'normal', #Inicializador dos pesos -> Random
                        input_dim = len(twenty_train))) #No de entradas


text_clf.add(Dense(units = 128, #No de entradas (30) + no saidas (1) / 2
                        activation = 'relu',
                        kernel_initializer = 'normal')) #Inicializador dos pesos -> Random
                       


text_clf.add(Dense(units = 128, #No de entradas (30) + no saidas (1) / 2
                        activation = 'relu',
                        kernel_initializer = 'normal', #Inicializador dos pesos -> Random
                        input_dim = 30)) #No de entradas


text_clf.add(Dense(units = 1, #Uma soh saida (true or false)
                        activation = 'sigmoid'))

import keras
otimizador = keras.optimizers.Adam(lr = 0.001, #Learning Rate
                                   decay = 0.0001,#Quanto que ele decrementa ao descer o gradiente
                                   clipvalue = 0.5) #Trava pesos, para n passar de 0.5

text_clf.compile(optimizer = otimizador, #Otimizador + utilizado. Se n funcionar, testar outros
                      loss = 'binary_crossentropy', #Calculo do erro para 1 saida True/False
                      metrics = ['binary_accuracy']) #Errado/Certo. Pode ter mais de uma metrica.

import numpy as np
text_clf = text_clf.fit(np.array(X_train_tfidf), np.array(twenty_train.target),
                  batch_size = 10, #Faz ajuste de pesos de 10 em 10
                  epochs = 100) #Qtas vezes o ajuste eh feito

'''Teste'''

import numpy as np
twenty_test = fetch_20newsgroups (subset='test', shuffle=True)
predicted_svm = text_clf.predict(twenty_test.data)
accuracy_SVM = np.mean(predicted_svm == twenty_test.target)

'''Daqui pra baixo sao codigos de teste'''

'''Grid Search'''

from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),}

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)

gs_clf.best_score_
gs_clf.best_params_

'''Refinamento dos Resultados'''

'''Stop Words''' #Eliminar palavras inuteis

#text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2),stop_words='english')), [...]

'''NLTK -> Classifica palavras em uma so; [fishing, fisher, fish...] => fish'''
import nltk
nltk.download()

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyser(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
    
stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

text_clf = Pipeline([('vect', stemmed_count_vect),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', alpha=0.001, random_state=42)),
])
