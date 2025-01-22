# -*- coding: utf-8 -*-
"""
Script pour detecter les fraudes bancaires à l'aide du Machine Learning 

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
flag_undersampling = 0 # Flag pour activer ou non le sous-échantilonnage
flag_randomforest = 1 # Flag pour activer ou non les Forêts Aléatoires
flag_linearSVC = 0 # Flag pour activer ou non la linearSVC
flag_research_rf = 0 # Flag pour activer ou non la recherche des meilleurs paramètres

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

# Séparation descripteurs/étiquettes
X = data.drop(columns=["Class"]) # Descripteurs
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
    print("Dimensions de la base d'entrainement : {X_train.shape}")
    print("Répartition des données :")
    print(y_train.value_counts())

##############################################################################
## Rechercher des meilleurs hyperparamètres
##############################################################################
if flag_research_rf == 1 :
    # Ensemble de paramètres à tester pour les forêts aléatoires afin d'en trouver les meilleurs
    params_rf = {
        "n_estimators" : [100],
        "criterion" : ("gini", "entropy"),
        "max_depth" : list(range(1,20)),
        "min_samples_split" : [2,3,4],
        "min_samples_leaf" : list(range(1,20))
        }
    # Résultat 22/01/2025 :
        # {'criterion': 'gini',
        #  'max_depth': 9,
        #  'min_samples_leaf': 1,
        #  'min_samples_split': 4,
        #  'n_estimators': 100}
        
    # Rechercher des meilleurs paramètres avec la méthode GridSearchCV
    clf_rf = RandomForestClassifier(random_state=random_state)
    clf_rf_cv = GridSearchCV(clf_rf, params_rf, scoring="accuracy", n_jobs=-1, verbose=1, cv=3)
    clf_rf_cv.fit(X_train, y_train)
    
    print("Meilleurs paramètres trouvés :")
    print(clf_rf_cv.best_params_)

##############################################################################
## CLASSIFICATION AVEC FORETS ALEATOIRES
##############################################################################
if flag_randomforest == 1 :
    # Par défaut #
    # clf_rf = RandomForestClassifier(random_state=random_state, max_depth=2) # Création d'un Classifieur Forêts Aléatoires
    
    # Avec les meilleurs paramètres trouvés avec GridSearchCV #
    # Liste des meilleurs paramètres :
    best_criterion = "gini"
    best_max_depth = 9
    best_min_samples_leaf = 1
    best_min_samples_split = 4
    best_n_estimators = 100
        
    clf_rf = RandomForestClassifier(
        random_state=random_state,
        criterion=best_criterion,
        max_depth=best_max_depth,
        min_samples_leaf=best_min_samples_leaf,
        min_samples_split=best_min_samples_split,
        n_estimators=best_n_estimators)
    
    clf_rf.fit(X_train, y_train) # On entraine le modèle sur les données d'entrainement
    
    msg = ("###########################\n"
           "       RANDOM FOREST\n"
           "###########################\n")
    print(msg)
    
    # Utilisation de ma fonction print_performance pour afficher quelques indices de performances de mon modèle
    print_performance(clf_rf, X_train, y_train, X_test, y_test)
    
    ConfusionMatrixDisplay.from_estimator(clf_rf, X_test, y_test)
    plt.show()
    
    RocCurveDisplay.from_estimator(clf_rf, X_test, y_test)
    plt.show()

##############################################################################
## CLASSIFICATION AVEC SVM
##############################################################################
if flag_linearSVC == 1 :
    
    clf_svc = LinearSVC(random_state=random_state) # Création d'un Classifieur SVM
    clf_svc.fit(X_train, y_train) # Entrainement du modèle
    
    msg = ("###########################\n"
           "         LINEAR SVC\n"
           "###########################\n")
    print(msg)
    
    # Affichage de quelques indices de performances
    print_performance(clf_svc, X_train, y_train, X_test, y_test)

