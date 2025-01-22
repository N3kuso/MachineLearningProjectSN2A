# -*- coding: utf-8 -*-
"""
Script pour detecter les fraudes bancaires à l'aide du Machine Learning 

@author: kooky
"""

##############################################################################
## CHARGEMENT & TRAITEMENT DES DONNEES
##############################################################################
import numpy as np #Calcul mathématiques
import matplotlib.pyplot as plt #Pour les tracés (alternative: from pylab import *)
import pandas as pd

## Importation des données ##
filename = "creditcard.csv"
data = pd.read_csv(filename, delimiter=',')

print(data)
print(f"Dimensions du dataset : {data.shape}")

# # Optionel
# import seaborn as sns

# sns.countplot(data=data, x="Class", hue="Class", stat="percent")
# plt.title("Nombre de transactions concernés par la fraude")
# plt.show()

## Traitement des données ##
from sklearn.model_selection import train_test_split

random_state = 42 # On fixe un random_state pour la répétabilité

# Séparation descripteurs/étiquettes
X = data.drop(columns=["Class"]) # Descripteurs
y = data["Class"] # Etiquettes

# Séparation du dataset en données d'entrainement et de test
train_size = 0.7
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)

# Affichage des dimensions
print(f"Dimensions de la base d'entrainement : {X_train.shape}")
print(f"Dimensions de la base de test : {X_test.shape}")

# (A FAIRE) Sous-échantilonné le dataset



##############################################################################
## CLASSIFICATION AVEC FORETS ALEATOIRES
##############################################################################
from sklearn.ensemble import RandomForestClassifier
from FunctionsCreditCardFraudDetection import print_performance

clf_rf = RandomForestClassifier(random_state=random_state, max_depth=2) # Création d'un Classifieur Forêts Aléatoires
clf_rf.fit(X_train, y_train) # On entraine le modèle sur les données d'entrainement

msg = ("###########################\n"
       "       RANDOM FOREST\n"
       "###########################\n")
print(msg)
# Utilisation de ma fonction print_performance pour afficher quelques indices de performances de mon modèle
print_performance(clf_rf, X_train, y_train, X_test, y_test)


##############################################################################
## CLASSIFICATION AVEC SVM
##############################################################################
from sklearn.svm import LinearSVC

clf_svc = LinearSVC(random_state=random_state) # Création d'un Classifieur SVM
clf_svc.fit(X_train, y_train) # Entrainement du modèle

msg = ("###########################\n"
       "         LINEAR SVC\n"
       "###########################\n")
print(msg)

# Affichage de quelques indices de performances
print_performance(clf_svc, X_train, y_train, X_test, y_test)

