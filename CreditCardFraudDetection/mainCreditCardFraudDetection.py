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

