'''
    there are two class can be used
    KKmeans()   : no cache version - kernel k-means
    KKmeans_cache() : cache version - kernel k-means
    
    ver.    update.         description.
    -------------------------------
    1.0     2019/04/12      
'''

import math
import numpy as np
import random

import time
from scipy.spatial import distance
import matplotlib.pyplot as plt



class KKmeans():
    ''' kernel K-means '''
    def __init__(self, k=3, initMethod='randomOriDis', iteration=300, kMethod='RBF', sigma=5, complement=0, degree=2):
        '''
            k : int( > 1), num of clusters
            initMethod : str('randomOriDis' or 'random'), init center method
            iteration : int( > 0), iteration times
            kMethod : str('RBF' or 'POLY'), kernel function method
            sigma : number(!= 0), arg of RBF kernel function, ref by _rbfKernel()
            complement :　number, arg of polynomial kernel function, ref by _polyKernel()
            degree : number, arg of polynomial kernel function, ref by _polyKernel()
        '''
        
        # args
        self.k = k # num of clusters
        self.initMethod = initMethod # init center method
        self.iter = iteration # iteration times
        self.kMethod = kMethod # kernel function method
        self.sigma = sigma # arg of RBF kernel function, ref by _rbfKernel()
        self.complement = complement # arg of polynomial kernel function, ref by _polyKernel()
        self.degree = degree # arg of polynomial kernel function, ref by _polyKernel()
        
        # attributes
        self.labels_ = None
        self.cluster_centers_ = None
    
    #=======================#
    #======= Methods =======#
    #=======================#
    def fit(self, X, k=None, initMethod=None, iteration=None):
        '''
            fit: fit the model
            @param X : a numpy array of all points
            @param k : int, num of cluster, default=None:ref by _run()
            @return : None
        ''' 
        X = self._normalize(X,X)
        self.labels_ = self._run(X, k=k, iteration=iteration, initMethod=initMethod)
    
    #=============================#
    #======= Inner Methods =======#
    #=============================#
    def _normalize(self, X, ref):
        ''' 
            min-max normalize := x - min / (max - min)
            @param X : 2D array, list of data
            @param ref : reference set to get min and max 
            normalize X by ref-min & ref-max along axis=0
        '''
        min_s = np.ndarray.min(ref, axis=0)
        max_s = np.ndarray.max(ref, axis=0)
        return (X - min_s) / (max_s-min_s)
    
    def _initial(self, X, k, initMethod='randomOriDis'):
        '''
            initial attributes
            @param X: 2D np array : all points
            @param k: int, num of clusters
            @param initMethod: for 'random', 'randomOriDis' :get init center method, default='randomOriDis': random center and get by origin Euchildean distance
            @return :1D array , init labels
        '''
        
        if initMethod == 'randomOriDis': # initial cluster by random center and count by origin distance
            # randomly initialize centroid
            cluster_centers_ = self._random_centroid(X, k)  
            # get initial labels
            labels = self._assignment_byCenter(cluster_centers_, X)#assign label by centroid
        elif initMethod == 'random': # initial cluster by random assign
            n = X.shape[0]
            labels = np.concatenate([np.arange(k), np.random.randint(k, size=(n-k))])
            np.random.shuffle(labels)
        return labels
    
    def _random_centroid(self, X, k):
        '''
            _random_centroid: randomly choose @num of centroids from @X
            @param X : a numpy array of all points
            @param k : number of centroid to choose
            @return : a list of index of @X, len=@num 
        '''
        idx = np.random.randint(len(X), size=k)
        return X[ idx, : ]
    
    def _assignment_byCenter(self, centroid, X):
        '''
            _assignment: main operation of k-means, assign points to "k" clusters
            @param centroid : a numpy array of centroid
            @param X : a numpy array of all points
            @return : the lables, numpy array
        '''
        distance = self._calculate_distance(centroid, X)
        return distance.argmin(axis=1)
    
    ### about distance
    def _calculate_distance(self, centroid, X):
        '''
            _calculate_distance: calculate the distance of each point to the centroid
            @param centroid : a numpy array of centroid
            @param X : a numpy array of all points
            @return  : a numpy array of distance, e.g. distance[point][centroid_number]
        '''
        distance = np.zeros( (len(X), len(centroid)) )
        distance = np.array([self._dis(X[idx,:],centroid) for idx in range(len(X))])
        return distance
        
    def _dis(self, x, y):
        '''
            _dis: calculate the Euchildean distance
            @param x : a single data, a dimension list of x coordinate 
            @param y : a single data, a dimension list of y coordinate
            @return : the euchilden distance
        '''
        return self._dis_Euchildean(x, y)
    
    def _dis_Euchildean(self, x, y):
        '''
            in new version x or y can be 2D np array
            @param x : a single data, a dimension list of x coordinate 
            @param y : a single data, a dimension list of y coordinate
            @return : the euchilden distance
        '''
        ##old version
        #return np.linalg.norm( (x-y) )
        ##new version to get distance for N to 1
        return np.linalg.norm( (x-y), axis=1)
        
    ### about assign
    def _assignment(self, data, labels, k):
        ''' 
            _assignment : get a cluster member list by labels
            @param data : 2D array, the origin points list
            @param labels : 1D array, the cluster label for each point data
            @param k : int, num of clusters
            @return : a set of 2D np array , list of each clusterMember
        '''
        assert (data.shape[0] == labels.shape[0]) , '_assignment , data.shape[0] != labels.shape[0]'
        listClusterMember = [[] for i in range(k)]
        for idx in range(data.shape[0]):
            listClusterMember[labels[idx]].append(data[idx,:])
            
        return listClusterMember
    
    ### about update
    def _update(self, listClusterMember, data):
        '''
            _update: update labels
            @param listClusterMember : a set of 2D np array , list of each clusterMember
            @param data : 2D np array,  all points
            @return : 1D np array, new labels of all points
        '''
        k = len(listClusterMember)
        N = data.shape[0]
        kernelResult = np.ndarray(shape=(N, 0))
        for i in range(k): # each cluster
            clusterMember = np.asarray(listClusterMember[i])
            # 1st part of error : K(Xa,Xa) is same in all compare , so it is not important
            E1 = 0
            # 2nd part of error : 2 * ai * K(Xa,Xb)
            E2 = np.array([self._get_E2(data[j,:], clusterMember) for j in range(N)]).reshape([N,1])
            # 3rd part of error : ai * aj * K(Xa,Xb)
            E3 = self._get_E3(clusterMember)
            # distance := E1 - E2 + E3
            dis = E1 - E2 + E3
            kernelResult = np.concatenate((kernelResult, dis), axis=1)
        labels = np.argmin(kernelResult, axis=1)
        labels = self._comfirm_Labels(labels, k)
        return labels
        
    def _get_E2(self, data, clusterMember):
        ''' 
            get the error out the cluster
            @param data : 1D np array, one point
            @param clusterMember : 2D np array, points in this cluster
            @return :float, the 2nd part of error in this cluster
        '''
        N = clusterMember.shape[0]
        output = np.sum(self._kernel(data, clusterMember))
        output /= N
        output *= 2
        return output
    
    def _get_E3(self, clusterMember):
        ''' 
            get the error in the cluster
            @param clusterMember : 2D np array, points in this cluster
            @return :float, the 3rd part of error in this cluster
        '''
        output = 0.
        N = clusterMember.shape[0]
        for i in range(N):
            output += np.sum(self._kernel(clusterMember, clusterMember[i,:]))
        output /= N**2
        return output
        
    def _comfirm_Labels(self, labels, k):
        '''
            to comfirm there are k sets in labels, avoid empty set
            @param labels : 1D np array , origin labels
            @param k : int, target num of sets
            @return : 1D np array , the labels( no empty set )
        '''
        n = labels.shape[0]
        _labels = labels
        _sets, _labels = np.unique(_labels, return_inverse=True)
        
        while len(_sets) < k:
            _labels[random.randint(0, n)] = k-1
            _sets, _labels = np.unique(_labels, return_inverse=True)
        return _labels
        
            
        
    ### about kernel
    def _kernel(self, X, Y, kMethod=None, sigma=None, complement = None, degree = None):
        '''
            kernel function
            @param X: 2D or 1D np array, list of points or one of point ( both X and Y cannot be 2D in the same time)
            @param Y: 1D or 2D np array, one of point or list of points ( both X and Y cannot be 2D in the same time)
            @param kMethod : string, for 'RBF', 'POLY': kernel function method default=None: use self.kMethod
            @param sigma : for 'RBF' , ref by rbfKernel()
            @param complement : for 'POLY' , ref by polyKernel()
            @param degree : for 'POLY' , ref by polyKernel()
            @return : 1D np array or one np value, K(X,Y) output 
        '''
        if kMethod is None:
            kMethod = self.kMethod
            
        if kMethod == 'RBF':
            return self._rbfKernel(X, Y, sigma)
        elif kMethod == 'POLY':
            return self._polyKernel(X, Y, complement, degree)
        else:
            assert False, 'unknown kernel method ' + str(kMethod)
    
    def _rbfKernel(self, X, Y, sigma = None):
        ''' 
            RBF kernel function (guassian kernel) 
            @param X: 2D or 1D np array, list of points or one of point ( both X and Y cannot be 2D in the same time)
            @param Y: 1D or 2D np array, one of point or list of points ( both X and Y cannot be 2D in the same time)
            @param sigma : float or int , the arg for rbf kernel function , default=None : use sigma from object config
            @return : 1D np array or one np value, K(X,Y) output 
        '''
        if sigma is None:
            sigma = 5 # use default
        delta = X - Y
        if len(delta.shape) < 2:
            squaredEuclidean = np.square(delta)
        else :
            squaredEuclidean = (np.square(delta).sum(axis=1))
        output = np.exp(-(squaredEuclidean)/(2*sigma**2))
        return output
    
    def _polyKernel(self, X, Y, complement = None, degree = None):
        ''' 
            Polynomial kernel function
            @param X: 2D or 1D np array, list of points or one of point ( both X and Y cannot be 2D in the same time)
            @param Y: 1D or 2D np array, one of point or list of points ( both X and Y cannot be 2D in the same time)
            @param complement : float or int , the arg for poly kernel function , default=None : use c from object config
            @param degree : float or int , the arg for poly kernel function , default=None : use c from object config
            @return : 1D np array or one np value, K(X,Y) output 
        '''
        if complement is None:
            complement = self.complement # use default
        if degree is None:
            degree = self.degree # use default
            
        product = np.matmul(X, Y.T)       
        output = np.power(product + complement, degree)
        return output
    
    ### about run
    def _run(self, X, iteration=None, k=None, initMethod=None):
        ''' 
            run the kernel k means 
            @param X : 2D np array, all points
            @param iteration : int, the total iteration times
            @param k : the num of cluster
            @param initMethod: initial method , default=None:use self.initMethod
            @return : 1D np array, the labels of all points
        '''
        # get args
        if k is None :
            k = self.k
        if initMethod is None:
            initMethod = self.initMethod
        if iteration is None:
            iteration = self.iter
            
        # randomly initialize labels
        labels = self._initial(X, k, initMethod)
        
        # main loop
        for i in range(iteration):
            print('  iter : ' + str(i))
            # Assignment
            listClusterMember = self._assignment(X, labels, k)

            # Update
            new_labels = self._update(listClusterMember, X)
            
            #if no change
            if (labels == new_labels).all():
                break
            else:
                labels = new_labels
               
        self.cluster_centers_ = np.array([np.mean(np.asarray(X[labels == i]), axis=0) for i in range(k)])
        return labels
        
        

class KKmeans_cache(KKmeans):
    ''' kernel K-means by cache '''
    def __init__(self, k=3, initMethod='randomOriDis', iteration=300, kMethod='RBF', sigma=5, complement=0, degree=2):
        '''
            k : int( > 1), num of clusters
            initMethod : str('randomOriDis' or 'random'), init center method
            iteration : int( > 0), iteration times
            kMethod : str('RBF' or 'POLY'), kernel function method
            sigma : number(!= 0), arg of RBF kernel function, ref by _rbfKernel()
            complement :　number, arg of polynomial kernel function, ref by _polyKernel()
            degree : number, arg of polynomial kernel function, ref by _polyKernel()
        '''
        super().__init__(k, initMethod, iteration, kMethod, sigma, complement, degree)
        
        self.kCache = None # np 2D array ,cache all K(xi,xj)


        # for calculating SSE 
        self.sseList = None 
        self.__convergeRatio = 0.7
        self.__atLeastIter = 15






    #=======================#
    #======= Methods =======#
    #=======================#
    def fit(self, X, k=None, initMethod=None, iteration=None):
        '''
            fit: fit the model
            @param X : a numpy array of all points
            @param k : int, num of cluster, default=None:ref by _run()
            @return : None
        ''' 
        X = X.astype(dtype=np.float32)
        #X = self._normalize(X,X)
        self.labels_ = self._run(X, k=k, iteration=iteration, initMethod=initMethod)
    
    #=============================#
    #======= Inner Methods =======#
    #=============================#
    def _normalize(self, X, ref):
        ''' 
            min-max normalize := x - min / (max - min)
            @param X : 2D array, list of data
            @param ref : reference set to get min and max 
            normalize X by ref-min & ref-max along axis=0
        '''
        return super()._normalize(X, ref)
    
    def _initial(self, X, k, initMethod='randomOriDis'):
        '''
            initial attributes
            @param X: 2D np array : all points
            @param k: int, num of clusters
            @param initMethod: for 'random', 'randomOriDis' :get init center method, default='randomOriDis': random center and get by origin Euchildean distance
            @return :1D array , init labels
        '''
        
        labels = super()._initial( X, k, initMethod='randomOriDis')
        
        return labels
        
    def _get_cache(self, data):
        n = data.shape[0]
        #cache = np.array([super()._kernel(data, data[i,:]) for i in range(n)])
        cache = np.ndarray(shape=(n,0))
        for i in range(n):
            temp = super()._kernel(data, data[i,:]).reshape((n,1))
            #print(str(cache.shape))
            cache = np.concatenate((cache, temp), axis=1)
        return cache
      
    ### about assign
    def _assignment(self, data, labels, k):
        ''' 
            _assignment : get a cluster member list by labels
            @param data : 2D array, the origin points data list
            @param labels : 1D array, the cluster label for each point data
            @param k : int, num of clusters
            @return : a set of 2D np array , list of each clusterMember(ids)
        '''
        assert (data.shape[0] == labels.shape[0]) , '_assignment , dataId.shape[0] != labels.shape[0]'
        listClusterMember = [[] for i in range(k)]
        for idx in range(data.shape[0]):
            listClusterMember[labels[idx]].append(idx)
        return listClusterMember
    
    ### about update
    def _update(self, listClusterMember, cache):
        '''
            _update: update labels
            @param listClusterMember : a set of 2D np array , list of each clusterMember ids
            @param cache : 2D np array,  k(i,j) for all points
            @return : 1D np array, new labels of all points
        '''
        k = len(listClusterMember)
        N = cache.shape[0]
        kernelResult = np.ndarray(shape=(N, 0))

        for i in range(k): # each cluster
            clusterMember = np.asarray(listClusterMember[i])
            # 1st part of error : K(Xa,Xa) is same in all compare , so it is not important
            E1 = 0
            # 2nd part of error : 2 * ai * K(Xa,Xb)
            E2 = np.array([self._get_E2(j, clusterMember) for j in range(N)]).reshape([N,1])
            # 3rd part of error : ai * aj * K(Xa,Xb)
            E3 = self._get_E3(clusterMember)
            # distance := E1 - E2 + E3
            dis = E1 - E2 + E3
            kernelResult = np.concatenate((kernelResult, dis), axis=1)
        
        labels = np.argmin(kernelResult, axis=1)
        labels = self._comfirm_Labels(labels, k)
        
        return labels
        
    def _get_E2(self, dataID, clusterMember):
        ''' 
            get the error out the cluster
            @param dataID : int , one point id
            @param clusterMember : 1D np array, points id in this cluster
            @return :float, the 2nd part of error in this cluster
        '''
        N = clusterMember.shape[0]
        output = np.sum(self._kernel(dataID, clusterMember))
        output /= N
        output *= 2
        return output
    
    def _get_E3(self, clusterMember):
        ''' 
            get the error in the cluster
            @param clusterMember : 1D np array, points id in this cluster
            @return :float, the 3rd part of error in this cluster
        '''
        output = 0.
        N = clusterMember.shape[0]
        for i in range(N):
            output += np.sum(self._kernel(clusterMember, clusterMember[i]))
        output /= N**2
        return output
        
    def _comfirm_Labels(self, labels, k):
        '''
            to comfirm there are k sets in labels, avoid empty set
            @param labels : 1D np array , origin labels
            @param k : int, target num of sets
            @return : 1D np array , the labels( no empty set )
        '''
        return super()._comfirm_Labels(labels, k)      
        
    ### about kernel
    def _kernel(self, X, Y):
        '''
            kernel function
            @param X: int or 1D np array, list of points or one of point ( both X and Y cannot be array in the same time)
            @param Y: 1D np array or int , one of point or list of points ( both X and Y cannot be array in the same time)
            @return : 1D np array or one np value, K(X,Y) output 
        '''
        return self.kCache[X,Y]
    
    ### about run
    def _run(self, X, iteration=None, k=None, initMethod=None):
        ''' 
            run the kernel k means 
            @param X : 2D np array, all points
            @param iteration : int, the total iteration times
            @param k : the num of cluster
            @param initMethod: initial method , default=None:use self.initMethod
            @return : 1D np array, the labels of all points
        '''
        # get args
        if k is None :
            k = self.k
        if initMethod is None:
            initMethod = self.initMethod
        if iteration is None:
            iteration = self.iter
        
        # initialize labels
        labels = self._initial(X, k, initMethod)
        
        X_norm = self._normalize(X, X)

        
        # get cache about k(xi,xj)
        self.kCache = self._get_cache(X_norm)
        
        # main loop
        for i in range(iteration):
            if (i % 10) == 0:
                print('  iter : ' + str(i))
            
            # Assignment
            listClusterMember = self._assignment(X, labels, k)

            
            # Update
            new_labels = self._update(listClusterMember, self.kCache)
            
            #if no change
            if (labels == new_labels).all():
                break
            else:
                labels = new_labels
            
            self.cluster_centers_ = np.array([np.mean(np.asarray(X[labels == i]), axis=0) for i in range(k)])
            
            
            # Calculate SSE, and test if the algorithm is converge
            error = self._sse(self.cluster_centers_, X, labels)
            print("Interation %d, SSE: %f" % (i+1, error))
            self.sseList = np.append(self.sseList, error)
            if( self._is_converge(self.sseList, self.__convergeRatio, self.__atLeastIter) ):
                print("Converge at Iteration %d, SSE: %f" % (i+1, error))
                break



        return labels













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

