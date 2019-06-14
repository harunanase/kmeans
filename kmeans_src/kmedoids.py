from kmeans import Kmeans
import numpy as np
from scipy.spatial import distance

class Kmedoids(Kmeans):

    def __init__(self, args, initCentroid=None):
        super().__init__(args, initCentroid)
        self.__parent = Kmeans
    




    """
        override methods from parent class - Kmeans
        _update: update the centroid
        @param centroid : a numpy array of centroid
        @param X : a numpy array of all points
        @param Y : a numpy array of labels
        @return : the new centroid of each "k" clusters

    """
    def _update(self, centroid, X, Y):

        clusterList = self._get_cluster_list(centroid, X, Y)
        newCentroid = [ c[self.__medoids_select(c)] for c in clusterList ]
       
        return newCentroid





    """
        __medoids_select: calculate the medoids of a cluster
        @param cluster : a numpy array of a cluster
        @return : the medoid its index in cluster array
    """
    def __medoids_select(self, cluster):
        
        distanceList = np.zeros(0)
        for i, idx in zip(cluster, range(len(cluster))):

            # remove current i points(index: idx) in cluster
            others = np.delete(cluster, idx, axis=0)

            # calculate distance
            distanceMethod = 'euclidean'
            current = i.reshape(1, len(i))
            total = distance.cdist(current, others, distanceMethod).sum()

            distanceList = np.append(distanceList, total)

        return distanceList.argmin()
        
