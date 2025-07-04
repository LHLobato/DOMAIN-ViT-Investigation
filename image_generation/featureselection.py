from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import io
from sklearn.pipeline import Pipeline
from transformers import BertModel, BertTokenizer
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler 
import torch
import pandas as pd
from sklearn.decomposition import PCA

import gc

domains = pd.read_csv("../datasets/dataset.csv")

print(domains.head())

domain_urls = domains['name'].values

dns = domains.drop(columns=['name','malicious'])

print(dns.shape, dns.head())

scaler = MinMaxScaler()

dns = scaler.fit_transform(dns)

labels = domains['malicious'].values

res = 64
features = [64,512, 1024, 2048, 4096]
for feature in features:
    vectorizer = TfidfVectorizer(analyzer="char", sublinear_tf=True,lowercase=False, ngram_range=(3,3), max_features=feature)
    X = vectorizer.fit_transform(domain_urls).toarray()
    X = scaler.fit_transform(X)

    data = np.hstack([X,dns])

    for state in [0,100]:
        X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=state
        )

        classifier = RandomForestClassifier(n_estimators=200)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_probs = classifier.predict_proba(X_test)[:, 1]  # Probabilidades da classe positiva
        test_auc = roc_auc_score(y_test, y_probs)
        print("-------FOR TFIDVECTORIZER-----------")
        print("AUC no conjunto de teste:", test_auc)
        print("Acurácia: ", accuracy_score(y_test, y_pred))
        print(f"Desempenho atingido com : {feature} features no TfidfVectorizer.")
        print(f"Random state do das divisões: {state}.")
        gc.collect()
