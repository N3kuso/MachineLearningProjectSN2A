# -*- coding: utf-8 -*-
"""
Librairie de fonctions utiles pour le script mainCreditCardFraudDetection.py

@author: kooky
"""

##############################################################################
## MODULES UTILES
##############################################################################
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd

##############################################################################
## FONCTIONS
##############################################################################
def print_performance(clf, x_train, y_train, x_test, y_test):
    # Train part
    print("Train Result:")
    print("======================================")
    score = accuracy_score(y_train, clf.predict(x_train))
    print(f"Accuracy Score: {round(score*100,2)} %")
    print("______________________________________")
    print("CLASSIFICATION REPORT:")
    clf_report = pd.DataFrame(classification_report(y_train, clf.predict(x_train), output_dict=True))
    print(clf_report)
    print("______________________________________")
    print("Confusion Matrix")
    matrix= confusion_matrix(y_train, clf.predict(x_train))
    print(matrix,"\n")
    
    #Test part
    print("Test Result:")
    print("======================================")
    score = accuracy_score(y_test, clf.predict(x_test))
    print(f"Accuracy Score: {round(score*100,2)} %")
    print("______________________________________")
    print("CLASSIFICATION REPORT:")
    clf_report = pd.DataFrame(classification_report(y_test, clf.predict(x_test), output_dict=True))
    print(clf_report)
    print("______________________________________")
    print("Confusion Matrix")
    matrix= confusion_matrix(y_test, clf.predict(x_test))
    print(matrix)