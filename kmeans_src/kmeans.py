import math
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt


class Kmeans():
    def __init__(self, args, initCentroid=None):
        
        # args
        self.k = args.cluster
        self.initMethod = args.init_method
        self.iter = args.iter

        # predefined constants
        self.__convergeRatio = args.converge_ratio
        self.__atLeastIter = args.at_least_iter


        # attributes
        self.labels_ = None
        self.initCentroid = initCentroid
        self.cluster_centers_ = None
        self.sseList = np.zeros(0)
        



    #=======================#
    #======= Methods =======#
    #=======================#
    """
        fit: fit the model
        @param X : a numpy array of all points
        @return : None
    """
    def fit(self, X):
        self.display_params()
        self.labels_ = self._run_kmeans(X, self.iter, self.initCentroid)      



    """
        display_params: display the model parameters
    """
    def display_params(self):
        print("===== Parameters ======")
        print("Name: ", self.__class__.__name__)
        print("k: ", self.k)
        print("initMethod: ", self.initMethod)
        print("initCentroid: ", self.initCentroid)
        print("iterations: ", self.iter)
        print("=======================")

    
    """
        plot_sse_chart: plot the sse chart
        @param chartName : the title of the chart
        @param path : save file path
    """
    def plot_sse_chart(self, chartName, path):
        plt.figure()
        plt.plot(self.sseList)
        plt.title(chartName)
        plt.ylabel('SSE')
        plt.xlabel('iterations')
        plt.savefig(path+chartName)


    



    #=============================#
    #======= Inner Methods =======#
    #=============================#
    """
        _dis: calculate the Euchildean distance
        @param x : a single data, a dimension list of x coordinate 
        @param y : a single data, a dimension list of y coordinate
        @return : the euchilden distance
    """
    def _dis(self, x, y):
        return np.linalg.norm( (x-y) )





    """
        _random_centroid: randomly choose @num of centroids from @X
        @param num : number of centroid to choose
        @param X : a numpy array of all points
        @return : a numpy array of k centroids
    """
    def _random_centroid(self, num, X):
        idx = np.random.randint(len(X), size=num)
        return X[ idx, : ]





    # author: 范真瑋
    def _kmeans_pp_centroid(self, k, X):       
        centroids = []
		# 隨機選一個樣本作為初始中心
        centroids.append(X[np.random.randint(0, len(X))])
        '''
        計算每個樣本和目前已有的中心的最短距離d
        並計算每個樣本被選為下一個中心的機率d / d.sum()
        最後依照該機率隨機選出下一個中心
        重複計算直到選出k個中心
        '''
        while len(centroids) < k:
            d = np.array([min([self._dis(i, c) for c in centroids]) for i in X])
            prob = d / d.sum()
            cum_prob = prob.cumsum() # 累積機率
			# 隨機產生0~1之間的數, 判斷屬於累積機率的哪個區間
            r = np.random.random()
			# 取得新中心的座標
            idx = np.where(cum_prob >= r)[0][0]
            centroids.append(X[idx])
        return np.array(centroids)
        """
        centroids = []
        centroids.append(X[np.random.randint(0, len(X))])
        while len(centroids) < k:
            d = np.array([min([self._dis(i, c) for c in centroids]) for i in X])
            prob = d / d.sum()
            cum_prob = prob.cumsum()
            r = np.random.random()
            idx = np.where(cum_prob >= r)[0][0]
            centroids.append(X[idx])
        return np.array(centroids)
        """








    """
        _calculate_distance: calculate the distance of each point to the centroid
        @param centroid : a numpy array of centroid
        @param X : a numpy array of all points
        @return  : a numpy array of labels, length = len(X)
    """
    def _calculate_distance(self, centroid, X):

        labels = np.zeros(0)
        distanceMethod = 'euclidean'
        for x in X:
            current = x.reshape(1, len(x))
            dis = distance.cdist(current, centroid, distanceMethod)
            labels = np.append(labels, np.argmin(dis))
        
        
        return labels





    """
        _get_cluster_list: get the cluster list of each centroid
        @param centroid : a numpy array of centroid
        @param X : a numpy array of all points
        @param Y : a numpy array of labels
        @return : the clusterList, LIST TYPE
    """
    def _get_cluster_list(self, centroid, X, Y):
        clusterList = [] # get cluster list
        clusterNum = len(centroid)
        for cnum in range(clusterNum):
            p, = np.where(Y == cnum)
            clusterList.append(X[p, :])
    
        return clusterList
    
   



    """
        _init: init the centroid of the algorithm
        @param k : the number of  cluster, integer
        @param X : a numpy of all points
        @return : a numpy array of k centroid
    """
    def _init(self, k, X, initCentroid):
        if initCentroid is not None:
            return initCentroid

        if self.initMethod == "random":
            # randomly initialize centroid
            return self._random_centroid(self.k, X)  
        else:
            # init method == kmean++
            return self._kmeans_pp_centroid(self.k, X)





    """
        _assignment: main operation of k-means, assign points to "k" clusters
        @param centroid : a numpy array of centroid
        @param X : a numpy array of all points
        @return : the labels, numpy array
    """
    def _assignment(self, centroid, X):
        labels =  self._calculate_distance(centroid, X)
        
        # check the empty cluster
        check = np.arange(self.k)
        for i in np.nditer(check):
            if( not (i in labels) ):   # stands for empty
                # empty, re-random centroid
                newCentroid = self._random_centroid(self.k, X)
                return self._assignment(newCentroid, X)
        return labels





    """
        _update: main operation of k-means, update the centroid in each new cluster
        @param centroid : a numpy array of centroid
        @param X : a numpy array of all points
        @param Y : a numpy array of labels
        @return : the new centroid of each "k" clusters
    """
    def _update(self, centroid, X, Y):

        clusterList = self._get_cluster_list(centroid, X, Y)
        newCentroid = [ np.mean(c, axis=0) for c in clusterList ]
        
        return newCentroid





    """
        _run_kmeans: the main routine of k-means
        @param X : a numpy array of all points
        @param iteration : the total iteration of k-means
        @return : the cluster result Y, that is the labels
    """
    def _run_kmeans(self, X, iteration, initCentroid=None):

        # init centroid
        self.initCentroid = self._init(self.k, X, initCentroid) if(initCentroid is None) else initCentroid
        self.cluster_centers_ = self.initCentroid
        


        # main loop
        for i in range(iteration):
            
            # Assignment
            self.labels_ = self._assignment(self.cluster_centers_, X)

            # Update
            self.cluster_centers_ = self._update(self.cluster_centers_, X, self.labels_)
            
            # Calculate SSE, and test if the algorithm is converge
            error = self._sse(self.cluster_centers_, X, self.labels_)
            print("Interation %d, SSE: %f" % (i+1, error))
            self.sseList = np.append(self.sseList, error)
            if( self._is_converge(self.sseList, self.__convergeRatio, self.__atLeastIter) ):
                print("Converge at Iteration %d, SSE: %f" % (i+1, error))
                break
            

        return self.labels_


    


    """
        _sse: calculate the sum of squard error
        @param centroid : a numpy array of centroid
        @param X : a numpy array of all points
        @param Y : a numpy array of labels
        @return : the total sse, float
    """
    def _sse(self, centroid, X, Y):
        clusterList = self._get_cluster_list(centroid, X, Y)
        distanceMethod = "sqeuclidean"
        total = 0
        for c, i in zip(centroid, clusterList):
            current = np.array(c).reshape(1, len(c))
            dis = distance.cdist(current, i, distanceMethod)
            total += dis.sum() 
            
        return total 





    """
        _is_converge: test if the algorithm is converge
        @param sseList : the numpy array of sse
        @param ratio: float, the ratio of the same element in array, it is used to determine the converge
        @param atLeastIter: integer, the smallset number of iteraions that the alogrithm should be run
    """
    def _is_converge(self, sseList, ratio, atLeastIter):
        convergeValue = sseList[-1] # get the last value 
        r, = np.where(convergeValue == sseList) # get the exist index array

        # if the value have more than the ratio(e.g. 2/3) of elements in the array, that means, converge
        # also, we should test it when the array size is > atLeastIter 
        return (len(r) > len(sseList)*ratio) and (len(sseList) > atLeastIter)

