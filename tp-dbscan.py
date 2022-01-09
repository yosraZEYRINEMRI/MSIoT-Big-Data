# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 20:58:51 2021

@author: huguet


"""

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator




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
#    2d-4c-no9.arff   xclara.arff
#  2d-4c-no4    spherical_4_3 
# cluto-t8-8k  cluto-t4-8k cluto-t5-8k cluto-t7-8k diamond9 banana
path = './artificial/'
databrut = arff.loadarff(open(path+"banana.arff", 'r'))
data = [[x[0],x[1]] for x in databrut[0]]
datanp = np.array([[x[0],x[1]] for x in databrut[0]])


########################################################################
# Preprocessing: standardization of data
########################################################################

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(datanp)

data_scaled = scaler.transform(datanp)

import scipy.cluster.hierarchy as shc

print("-------------------------------------------")
print("Affichage données standardisées            ")
f0_scaled = data_scaled[:,0] # tous les élements de la première colonne
f1_scaled = data_scaled[:,1] # tous les éléments de la deuxième colonne
#print(f0)
#print(f1)

plt.scatter(f0_scaled, f1_scaled, s=8)
plt.title("Donnees standardisées")
plt.show()

########################################################################
# Run DBSCAN clustering method 
# for a given number of parameters eps and min_samples
#  
distance=3
min_pts=5
cl_pred = cluster.DBSCAN(eps=distance, min_samples=min_pts).fit_predict(data)

# Plot results
plt.scatter(f0_scaled, f1_scaled, c=cl_pred, s=8)
plt.title(f"Clustering DBSCAN - Epilson={distance} - Minpt={min_pts}")
plt.show()
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(cl_pred)) - (1 if -1 in cl_pred else 0)
n_noise_ = list(cl_pred).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
if len(set(cl_pred)) > 1:
    silh = metrics.silhouette_score(datanp, cl_pred, metric='euclidean')
    print("Coefficient de silhouette : ", silh)


# Another example
distance=0.01
min_pts=5
cl_pred = cluster.DBSCAN(eps=distance, min_samples=min_pts).fit_predict(data)

# Plot results
plt.scatter(f0_scaled, f1_scaled, c=cl_pred, s=8)
plt.title(f"Clustering DBSCAN - Epilson={distance} - Minpt={min_pts}")
plt.show()
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(cl_pred)) - (1 if -1 in cl_pred else 0)
n_noise_ = list(cl_pred).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
if len(set(cl_pred)) > 1:
    silh = metrics.silhouette_score(datanp, cl_pred, metric='euclidean')
    print("Coefficient de silhouette : ", silh)

# Another example
distance=0.02
min_pts=5
cl_pred = cluster.DBSCAN(eps=distance, min_samples=min_pts).fit_predict(data)

# Plot results
plt.scatter(f0_scaled, f1_scaled, c=cl_pred, s=8)
plt.title(f"Clustering DBSCAN - Epilson={distance} - Minpt={min_pts}")
plt.show()
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(cl_pred)) - (1 if -1 in cl_pred else 0)
n_noise_ = list(cl_pred).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
if len(set(cl_pred)) > 1:
    silh = metrics.silhouette_score(datanp, cl_pred, metric='euclidean')
    print("Coefficient de silhouette : ", silh)

# Finding the optimal epsilon value

neigh = NearestNeighbors(n_neighbors=min_pts)
nbrs = neigh.fit(data_scaled)
distances, indices = nbrs.kneighbors(data_scaled)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.title("Nearest Neighbors : distance and associated indices")
plt.show()

kn = KneeLocator(distances, range(len(distances)), curve='convex', direction='increasing')
eps_opt = kn.knee

# Another example
distance= eps_opt
min_pts=5
cl_pred = cluster.DBSCAN(eps=distance, min_samples=min_pts).fit_predict(data)

# Plot results
plt.scatter(f0_scaled, f1_scaled, c=cl_pred, s=8)
plt.title(f"Clustering DBSCAN - Epilson={distance} - Minpt={min_pts}")
plt.show()
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(cl_pred)) - (1 if -1 in cl_pred else 0)
n_noise_ = list(cl_pred).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
if len(set(cl_pred)) > 1:
    silh = metrics.silhouette_score(datanp, cl_pred, metric='euclidean')
    print("Coefficient de silhouette : ", silh)


########################################################################
# FIND "interesting" values of epsilon and min_samples 
# using distances of the k NearestNeighbors for each point of the dataset
#
# Note : a point x is considered to belong to its own neighborhood  


