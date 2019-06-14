from sklearn.cluster import KMeans
from sklearn.metrics.cluster import completeness_score
from sklearn import datasets


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


import kmeans as mykm
import kmedoids as mykmedoids
import kmedians as mykmedians
import kkmeans as mykkmeans


"""
    Base class
"""
class Dataset():
    def __init__(self, args):

        # args
        self.k = args.cluster
        self.initMethod = args.init_method
        self.iter = args.iter
        
        self.dataset = None
        self.featureName = None
        self.X = None
        self.Y = None


    def plot_label_chart(self, title, xName, yName, Y, save=False, path=None):
        plt.hist(Y, bins=self.k, alpha=0.7)
        plt.title(title)
        plt.ylabel(yName)
        plt.xlabel(xName)

        plt.xlim(0, self.k)
        plt.xticks(range(self.k), [ i for i in range(self.k)])
        plt.savefig(path+title)



    def plot_result(self, title, xName, yName, zName, X, Y, Z, C, save=False, path=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = X
        y = Y
        z = Z
        c = C

        ax.set_title(title)
        ax.set_xlabel(xName)
        ax.set_ylabel(yName)
        ax.set_zlabel(zName)

        ax.scatter(x, y, z, c=c)
            
        if(save):
            plt.savefig(path+title)
        else:
            self.show_plot_chart()



    def show_plot_chart(self):
        plt.show()


       
    """
        check out the below references
    """
    def accuracy(self, groundTruth, predict):
        # Reference [1]: https://stackoverflow.com/questions/51320227/determining-accuracy-for-k-means-clustering 
        # Reference [2]: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html#sklearn.metrics.completeness_score 
        return completeness_score(groundTruth, predict)





    def do_sklearn_Kmeans(self):
        
        km = KMeans(n_clusters=self.k, init=self.initMethod, max_iter=self.iter)

        print("fitting the model (sklearn) ...")
        start_time = time.time()
        start_clock = time.clock()
        km.fit(self.X)
        print("--- time: %s seconds\t clock: %s seconds ---" % (time.time() - start_time, time.clock() - start_clock))

        print("acc: ", self.accuracy(self.Y, km.labels_))
        print("===== K-means cluster center (sklearn) =====")
        print(km.cluster_centers_)        
        print("============================================\n\n\n\n")
        
        return km


    def do_my_Kmeans(self, args, initCentroid=None):
        km = mykm.Kmeans(args, initCentroid=initCentroid)


        print("fitting the model (my own k-means) ...")
        start_time = time.time()
        start_clock = time.clock()
        km.fit(self.X)
        print("--- time: %s seconds\t clock: %s seconds ---" % (time.time() - start_time, time.clock() - start_clock))


        km.plot_sse_chart('my_kmeans_sse', args.save_path)

        print("acc: ", self.accuracy(self.Y, km.labels_))
        print("===== K-means cluster center (my own k-means) =====")
        print(km.cluster_centers_)        
        print("===================================================\n\n\n\n")
        
        return km



    def do_my_Kmedoids(self, args, initCentroid=None):
        km = mykmedoids.Kmedoids(args, initCentroid=initCentroid)

        print("fitting the model (my own k-medoids) ...")
        start_time = time.time()
        start_clock = time.clock()
        km.fit(self.X)
        print("--- time: %s seconds\t clock: %s seconds ---" % (time.time() - start_time, time.clock() - start_clock))

        km.plot_sse_chart('my_kmedoids_sse', args.save_path)

        print("acc: ", self.accuracy(self.Y, km.labels_))
        print("===== K-medoids cluster center (my own k-medoids) =====")
        print(km.cluster_centers_)        
        print("===================================================\n\n\n\n")

        return km


    def do_my_Kmedians(self, args, initCentroid=None):
        km = mykmedians.Kmedians(args, initCentroid=initCentroid)


        print("fitting the model (my own k-median) ...")
        start_time = time.time()
        start_clock = time.clock()
        km.fit(self.X)
        print("--- time: %s seconds\t clock: %s seconds ---" % (time.time() - start_time, time.clock() - start_clock))


        km.plot_sse_chart('my_kmedians_sse', args.save_path)


        print("acc: ", self.accuracy(self.Y, km.labels_))
        print("===== K-median cluster center (my own k-median) =====")
        print(km.cluster_centers_)        
        print("===================================================\n\n\n\n")

        return km


    def do_my_KKmeans(self, args, initCentroid=None, kMethod='RBF'):
        km = mykkmeans.KKmeans_cache(k=self.k, initMethod=self.initMethod, iteration=self.iter, kMethod=kMethod)


        print("fitting the model (my own k-kmeans) ...", km.kMethod)
        start_time = time.time()
        start_clock = time.clock()
        km.fit(self.X)
        print("--- time: %s seconds\t clock: %s seconds ---" % (time.time() - start_time, time.clock() - start_clock))


        km.plot_sse_chart('my_kkmeans_sse', args.save_path)


        print("acc: ", self.accuracy(self.Y, km.labels_))
        print("===== K-kmeans cluster center (my own k-kmeans) =====")
        print(km.cluster_centers_)        
        print("===================================================\n\n\n\n")


        return km



class Iris_Kmeans(Dataset):
    def __init__(self, args):

        super().__init__(args)
        self.__parent = super()
               
    def load(self):
        # load dataset
        self.dataset = datasets.load_iris()
        self.featureName = self.dataset.feature_names
        self.X = self.dataset.data
        self.Y = self.dataset.target
        
        # display features name 
        print("Using features: ", self.featureName)
        
       
    
class Abalone_Kmeans(Dataset):
    def __init__(self, args):
        
        super().__init__(args)
        self.__parent = super()
        self.__ringsLength = 29



    def load(self, filename):       
        
        # read csv 
        self.__csv = filename
        self.__csvDataFrame = pd.read_csv(self.__csv, sep=',') 
               
        self.dataset = self.__csvDataFrame.values # orgin values

        self.featureName = self.__csvDataFrame.drop('rings', axis=1).columns
       
        self.X = self.__csvDataFrame.drop('rings', axis=1) # drop rings
        self.X = self.X.drop('sex', axis=1).values # drop sex
        self.Y = self.__csvDataFrame['rings'].values  # ONLY rings

        # mapping sex value
        #for i in np.nditer(self.X[:, 0], op_flags=['readwrite'], flags=['refs_ok']):
        #    i[...] = self.__sex_mapping(i)
            

        # mapping result Y 
        for i in np.nditer(self.Y, op_flags=['readwrite'], flags=['refs_ok']):
            i[...] = self.__result_mapping(i, self.k)


        # display features name 
        print("Using features: ", self.featureName)
        print(self.Y)        


    def __sex_mapping(self, x):
        if(x == 'M'):
            return 1
        elif(x == 'F'):
            return 2
        elif(x == 'I'):
            return 0
        else:
            return "unknown"

    

    def __result_mapping(self, x, k):
        needToAdd = k - (self.__ringsLength % k)
        origin = np.arange(1, self.__ringsLength+1)
        new = np.append(origin, [-1 for i in range(needToAdd)])
        new = new.reshape(k, int(new.size/k))
        
        i, j = np.where(new == x)
        return i



class GeneralDS(Dataset):
    def __init__(self, args):
        super().__init__(args)
        self.__parent = super()

    
    def load(self, filename):
        self.__csv = filename
        self.__csvDataFrame = pd.read_csv(self.__csv, sep=',')

        self.dataset = self.__csvDataFrame.values
        self.featureName = self.__csvDataFrame.columns
  
        self.X = self.dataset
        
        # display features name 
        print("Using features: ", self.featureName)




