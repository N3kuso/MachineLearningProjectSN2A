# -*- coding: utf-8 -*-
"""
Script pour detecter les fraudes bancaires à l'aide du Machine Learning 

@author: kooky
"""

##############################################################################
## IMPORTATION DES LIBRAIRIES
##############################################################################
import numpy as np #Calcul mathématiques
import matplotlib.pyplot as plt #Pour les tracés (alternative: from pylab import *)
import pandas as pd

##############################################################################
## CHARGEMENT DES DONNEES
##############################################################################
filename = "creditcard.csv"
data = pd.read_csv(filename, delimiter=',')

print(data)
print(f"Dimensions du dataset : {data.shape}")

# # Optionel
# import seaborn as sns

# sns.countplot(data=data, x="Class", hue="Class", stat="percent")
# plt.title("Nombre de transactions concernés par la fraude")
# plt.show()

##############################################################################
## 1er Test Arbre de décision
##############################################################################
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Séparation descripteurs/étiquettes
X = data.drop(columns=["Class"]) # Descripteurs
y = data["Class"] # Etiquettes

random_state = 42
clf = DecisionTreeClassifier(criterion="entropy", max_depth=20, random_state=random_state)
clf.fit(X, y)

clf_score_train = clf.score(X, y)

print(f"Score pour base d'entrainement : {clf_score_train}")

tree.plot_tree(clf, feature_names=list(X.columns))
plt.show()

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(clf, X, y)
plt.show()