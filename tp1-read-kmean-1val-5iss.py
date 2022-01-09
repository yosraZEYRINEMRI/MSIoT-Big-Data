# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 23:08:23 2021

@author: huguet
"""
import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

##################################################################
# READ a data set (arff format)

# Parser un fichier de données au format arff
# datanp est un tableau (numpy) d'exemples avec pour chacun la liste 
# des valeurs des features

# Note 1 : 
# dans les jeux de données considérés : 2 features (dimension 2 seulement)
# t =np.array([[1,2], [3,4], [5,6], [7,8]]) 
#
# Note 2 : 
# le jeu de données contient aussi un numéro de cluster pour chaque point
# --> IGNORER CETTE INFORMATION ....
#    2d-4c-no9.arff

path = './artificial/'
databrut = arff.loadarff(open(path+"banana.arff", 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])
#print(databrut)
#print(datanp)

##################################################################
# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            ")
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales")
plt.show()

##################################################################
# Run clustering method for a given number of clusters
print("------------------------------------------------------")
print("Appel KMeans pour une valeur de k fixée (données init)")
tps1 = time.time()
k=5
model_km = cluster.KMeans(n_clusters=k, init='k-means++')
model_km.fit(datanp)
tps2 = time.time()
labels_km = model_km.labels_
# Nb iteration of this method
iteration = model_km.n_iter_

# Résultat du clustering
plt.scatter(f0, f1, c=labels_km, s=8)
plt.title("Données (init) après clustering")
plt.show()
print("nb clusters =",k,", nb iter =",iteration, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")
#print("labels", labels_km)
# Some evaluation metrics
# inertie = wcss : within cluster sum of squares
inert = model_km.inertia_
silh = metrics.silhouette_score(datanp, model_km.labels_, metric='euclidean')
print("Inertie : ", inert)
print("Coefficient de silhouette : ", silh)

########################################################################
# TESTER PARAMETRES METHODE ET RECUPERER autres métriques
########################################################################

##################################################################
# Searching for optimal number of clusters based on silhouette metric

print("------------------------------------------------------")
print("Liste des k / n_clusters à tester")
k_values = range(2, 11)
print(k_values)

durations = []
n_iters = []
metric_inert = []
metric_silh = []
labels_kms = []

for k in k_values:
    tps1 = time.time()
    model_km = cluster.KMeans(n_clusters=k, init='k-means++')
    model_km.fit(datanp)
    tps2 = time.time()
    labels_kms.append(model_km.labels_)
    
    # Nb iteration of this method
    iteration = model_km.n_iter_
    duration = round((tps2 - tps1)*1000,2)
    inert = model_km.inertia_
    silh = metrics.silhouette_score(datanp, model_km.labels_, metric='euclidean')
    
    durations.append(duration)
    n_iters.append(iteration)
    metric_inert.append(inert)
    metric_silh.append(silh)
    
plt.plot(k_values, metric_silh)
plt.title("Silhouette en fct de n_clusters")
plt.show()

plt.plot(k_values, metric_inert)
plt.title("Inertie en fct de n_clusters")
plt.show()

plt.plot(k_values, durations)
plt.title("Durée d'exec en fct de n_clusters")
plt.show()

plt.plot(k_values, n_iters)
plt.title("N_iter en fct de n_clusters")
plt.show()

# Résultat du clustering

# looking for index maximizing silhouette
max_silh = max(metric_silh)
opt_idx = metric_silh.index(max_silh)
k_opt = k_values[opt_idx]
labels_km = labels_kms[opt_idx]

iterations = n_iters[opt_idx]
duration = durations[opt_idx]
inert = metric_inert[opt_idx]
silh = metric_silh[opt_idx]



plt.scatter(f0, f1, c=labels_km, s=8)
plt.title("Données (init) après clustering")
plt.show()
print("nb clusters =",k_opt,", nb iter =",iterations, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")
#print("labels", labels_km)
# Some evaluation metrics
# inertie = wcss : within cluster sum of squares
print("Inertie : ", inert)
print("Coefficient de silhouette : ", silh)
