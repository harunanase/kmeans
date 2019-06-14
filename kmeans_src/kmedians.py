import math
import numpy as np
import kmeans 

class Kmedians(kmeans.Kmeans):
    def __init__(self, args, initCentroid=None):
        
        # args -- invoke parent __init__
        super().__init__(args, initCentroid)


    #=======================#
    #======= Methods =======#
    #=======================#

    #=============================#
    #======= Inner Methods =======#
    #=============================#
    """
        CHOU updated: 
        _dis: calculate Manhattan distance
        @param x : a single data, a dimension list of x coordinate 
        @param y : a single data, a dimension list of y coordinate
        @return : the Manhattan distance
    """
    def _dis(self, x, y):
        return np.abs(x - y).sum()

    
    """
        CHOU updated: use medians instead
        __update: main operation of k-medians, update the centroid in each new cluster
        @param centroid : a numpy array of centroid
        @param X : a numpy array of all points
        @param Y : a numpy array of labels
        @return : the new centroid of each "k" clusters
    """
    def _update(self, centroid, X, Y):
        clusterList = self._get_cluster_list(centroid, X, Y)
        newCentroid = []
        for c in clusterList:
            newCentroid.append(np.median(c, axis=0))

        return newCentroid

