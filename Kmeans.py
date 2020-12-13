#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:31:01 2020

@author: curtiswright
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt



def cluster(clusters,data):
    #create a dataframe of random cluster centers 
    centroids = pd.DataFrame(np.random.rand(clusters,data.shape[1]), columns=list(data.columns.values))
    
    #create a distance matrix of points to centroids
    distance = pd.DataFrame(np.zeros((data.shape[0],clusters),dtype = 'float32'))
    
    for i in range(data.shape[0]):
        for j in range(clusters):
             squares = (data.loc[i]-centroids.loc[j])**2
             distance.at[i,j] = (squares.sum(axis = 0))**1/2
    
    #assign clusters
    assignment = distance.idxmin(axis=1, skipna=True)
 
    #loop until distances stop changing
    converged = False
    
    counter = 0
    while not converged:
        
        counter+=1
        #count how many points are assigned to each centroid
        counts = pd.DataFrame(assignment.value_counts()).sort_index()
        
        #dataframe new centroids
        Newcentroid = pd.DataFrame(np.zeros((clusters,data.shape[1]),dtype = 'float32'), columns=list(data.columns.values))
        
        #find new centers
        for k in range(clusters):
            for l in range(len(assignment)):
                if assignment[l] == k:
                    ave = data.iloc[l,] / counts.at[k,0]
                    for m in data.columns.values:
                        Newcentroid.at[k,m] += ave[m]
       
        #check if centers moved
        if(Newcentroid - centroids).max().max() == 0:
            converged = True
        else:
            centroids = Newcentroid.copy()
            
        #recalculate distances
        for i in range(data.shape[0]):
            for j in range(clusters):
                 squares = (data.loc[i]-centroids.loc[j])**2
                 distance.at[i,j] = (squares.sum(axis = 0))**1/2
    
        #assign clusters
        assignment = distance.idxmin(axis=1, skipna=True)

    return(centroids,assignment,counter)



#import and normalize data  
data = pd.read_csv('A.csv')
clusters = 3
datanorm = (data-data.min())/(data.max()-data.min())

#call the function - note this returns normalized data.
centroids,assignment,counter = cluster(clusters,datanorm)

#unnoralize centroids
centroid_unnorm = centroids*(data.max()-data.min())+data.min()
data['cluster'] = assignment

#re-calculate distance matrix with unnormalized data.
distance = pd.DataFrame(np.zeros((data.shape[0],clusters),dtype = 'float32'))
for i in range(data.shape[0]):
    for j in range(clusters):
        squares = (data.loc[i]-centroid_unnorm.loc[j])**2
        distance.at[i,j] = (squares.sum(axis = 0))

#calculate Distortion factor
S = 0
for x in range(data.shape[0]):
    S += distance.at[x,assignment[x]]
    

#plot results
plt.scatter(data.iloc[:,0], data.iloc[:,1], c = assignment)
plt.scatter(centroid_unnorm.iloc[:, 0], centroid_unnorm.iloc[:, 1], s=300, c='red')
plt.show()

