# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 14:59:34 2025

@author: kooky
"""

##############################################################################
## LIBRAIRIES
##############################################################################
# Librairies principales
import numpy as np #Calcul mathématiques
import matplotlib.pyplot as plt #Pour les tracés (alternative: from pylab import *)
import pandas as pd
# Librairies pour le traitement des données
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
# Librairie pour Forêts Aléatoires
from sklearn.ensemble import RandomForestClassifier
# Librairies pour SVM
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
# Librairies de fonctions utiles
from FunctionsCreditCardFraudDetection import print_performance
# Librairies pour maximiser les performances
from sklearn.model_selection import GridSearchCV
# Librairies pour afficher les performances
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay

##############################################################################
## FLAG
##############################################################################
flag_undersampling = 1 # Flag pour activer ou non le sous-échantilonnage
flag_randomforest = 0 # Flag pour activer ou non les Forêts Aléatoires
flag_linearSVC = 0 # Flag pour activer ou non la linearSVC
flag_SVC = 0 # Flag pour activer ou non la SVC
flag_research_rf = 0 # Flag pour activer ou non la recherche des meilleurs paramètres pour RandomForest
flag_research_linearSVC = 0 # Flag pour activer ou non la recherche des meilleurs paramètres pour LinearSVC
flag_research_SVC = 0 # Flag pour activer ou non la recherche des meilleurs paramètres pour SVC

##############################################################################
## CHARGEMENT & TRAITEMENT DES DONNEES
##############################################################################
## Importation des données ##
filename = "creditcard.csv"
data = pd.read_csv(filename, delimiter=',')

# print(data)
print(f"Dimensions du dataset : {data.shape}")

# # Optionel
# import seaborn as sns

# sns.countplot(data=data, x="Class", hue="Class", stat="percent")
# plt.title("Nombre de transactions concernés par la fraude")
# plt.show()

## Traitement des données ##

random_state = 77 # On fixe un random_state pour la répétabilité



### Reduction de dimensions des CP ###
# Selection des colonnes de V1 à V28
acp_columns = [f"V{i}" for i in range(1, 29)]
acp_data = data[acp_columns]

# Calculer la variance pour chaque composante principale
variances = acp_data.var(axis=0)
variance_explained = variances / variances.sum()
cumulative_variance = np.cumsum(variance_explained)

# Visualiser la variance expliquée et la variance cumulée
plt.figure(figsize=(10, 6))
plt.bar(range(1, 29), variance_explained, alpha=0.7, label='Variance expliquée individuelle')
plt.step(range(1, 29), cumulative_variance, where='mid', label='Variance cumulée', color='red')
plt.xticks(range(1, 29))
plt.xlabel('Composantes principales')
plt.ylabel('Proportion de la variance expliquée')
plt.title('Analyse de la variance expliquée par les composantes principales')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Calcul de la corrélation de chaque CP avec 'class' en supprimant la dernière ligne correspondant à la correlation de 'class' avec 'class'
correlations = data[acp_columns + ['Class']].corr()['Class'].iloc[:-1]

correlations_sorted = correlations.sort_values(ascending=False)

best_correlations_columns = correlations_sorted.head(10).index.tolist() + correlations_sorted.tail(10).index.tolist()

# Séparation descripteurs/étiquettes
columns_list = best_correlations_columns + ["Time", "Amount"]
X = data[columns_list] # Descripteurs
y = data["Class"] # Etiquettes

# Séparation du dataset en données d'entrainement et de test
train_size = 0.7
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)

# Affichage des dimensions
print(f"Dimensions de la base d'entrainement : {X_train.shape}")
print(f"Dimensions de la base de test : {X_test.shape}")


if flag_undersampling == 1 :
    # Sous-échantilonner le dataset
    
    # Utilisation d'une méthode pour sous-échantilonner la classe Transaction Légitime (Majoritaire)
    rus = RandomUnderSampler(random_state=random_state)
    X_train, y_train = rus.fit_resample(X_train, y_train)
    print("Sous échantillonnage")
    print(f"Dimensions de la base d'entrainement : {X_train.shape}")
    print("Répartition des données :")
    print(y_train.value_counts())
    
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=random_state)
clf.fit(X_train, y_train)

# Utilisation de ma fonction print_performance pour afficher quelques indices de performances de mon modèle
print_performance(clf, X_train, y_train, X_test, y_test)

ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plt.show()

RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.show()