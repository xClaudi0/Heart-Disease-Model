#!/usr/bin/env python
# coding: utf-8
# Setuppo i dati
import pandas as pd
import numpy as np
heart_disease = pd.read_csv("/heart-disease.csv")
heart_disease
# Features matrix
x = heart_disease.drop("target", axis=1)
# Labels
y = heart_disease["target"]
# Scelta del modello e hyperparameters
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.get_params()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
clf.fit(x_train, y_train);
# Prediction
y_preds = clf.predict(x_test)
y_preds
y_test
# Valutazione modello sui train data
clf.score(x_train, y_train)
#Valutazione modello sui test data
clf.score(x_test, y_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_test, y_preds))
confusion_matrix(y_test, y_preds)
accuracy_score(y_test, y_preds)







