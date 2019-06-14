from argparse import ArgumentParser

from ds import *



def argument_parsing():
    parser = ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="all", help="kmeans, kmedoids, kmedians, kernel-kmeans")
    parser.add_argument("--cluster", type=int, default=3, help="the cluster number, a.k.a. K")
    parser.add_argument("--init-method", type=str, default="random", 
                            help="the init method of centroid selection, please give 'random' or 'k-means++'")
    parser.add_argument("--iter", type=int, default=300, help="the iteration of the k means")
    parser.add_argument("--dataset", type=str, default="iris", help="testing dataset, iris / abalone / general\n" + 
                                    "if general add --file=YOUR_FILE to give datasets csv file.")
    parser.add_argument("--file", type=str, default=None, help="the general dataset csv file name")
    parser.add_argument("--visual", type=str, default="save", help="visualizing the result, save / display the image")
    parser.add_argument("--save-path", type=str, default="./result_img/", help="the path where visualizing images saved")
    parser.add_argument("--kernel-method", type=str, default="RBF", help="the kernel k means method, RBF / POLY")
    parser.add_argument("--converge-ratio", type=float, default=0.7, help="the converge ratio")
    parser.add_argument("--at-least-iter", type=int, default=15, help="the minimun number to iterate")


    args = parser.parse_args()

    print("===== Arguments ======")
    for arg in vars(args):
        print(arg, ": ", getattr(args, arg))
    print("======================\n\n")

    return args


def do_iris(args):
    
    iris = Iris_Kmeans(args)

    # load dataset
    iris.load()
    
    # constant setting
    visualType = True if(args.visual == "save") else False
    X = iris.X
    Y = iris.Y

    if args.algorithm == 'all':
        # predict
        km_sklearn = iris.do_sklearn_Kmeans()
        km_myownkm = iris.do_my_Kmeans(args)
        km_myownkmedoids = iris.do_my_Kmedoids(args, initCentroid=km_myownkm.initCentroid)
        km_myownkmedians = iris.do_my_Kmedians(args, initCentroid=km_myownkm.initCentroid)
        km_myownkkmeans = iris.do_my_KKmeans(args, initCentroid=km_myownkm.initCentroid, kMethod=args.kernel_method)

        print("Saving result images...")
        iris.plot_result("iris-Ground_Truth", "petal width", "sepat lenth", "petal width", X[:, 0], X[:, 2], X[:, 3],
                            Y, save=visualType, path=args.save_path)
        iris.plot_result("iris-sklearn_kmeans", "petal width", "sepat lenth", "petal width", X[:, 0], X[:, 2], X[:, 3],
                            km_sklearn.labels_, save=visualType, path=args.save_path)
        iris.plot_result("iris-my_kmeans", "petal width", "sepat lenth", "petal width", X[:, 0], X[:, 2], X[:, 3],
                            km_myownkm.labels_, save=visualType, path=args.save_path)
        iris.plot_result("iris-my_kmedoids", "petal width", "sepat lenth", "petal width", X[:, 0], X[:, 2], X[:, 3],
                            km_myownkmedoids.labels_, save=visualType, path=args.save_path)
        iris.plot_result("iris-my_kmedians", "petal width", "sepat lenth", "petal width", X[:, 0], X[:, 2], X[:, 3],
                            km_myownkmedians.labels_, save=visualType, path=args.save_path)
        iris.plot_result("iris-my_kernel-kmeans", "petal width", "sepat lenth", "petal width", X[:, 0], X[:, 2], X[:, 3],
                            km_myownkkmeans.labels_, save=visualType, path=args.save_path)


    else:
        funcdict = {    'kmeans': iris.do_my_Kmeans, 'kmedoids': iris.do_my_Kmedoids,
                        'kmedians': iris.do_my_Kmedians, 'kernel-kmeans': iris.do_my_KKmeans    }
        km_mykm = funcdict[args.algorithm] (args) if args.algorithm != 'kernel-kmeans' else funcdict[args.algorithm] (args, kMethod=args.kernel_method) 

        print("Saving result images...")
        iris.plot_result('iris-my_'+args.algorithm, "petal width", "sepat lenth", "petal width", X[:, 0], X[:, 2], X[:, 3],
                            km_mykm.labels_, save=visualType, path=args.save_path)



def do_abalone(args):

    aba = Abalone_Kmeans(args)
    
    # load dataset
    aba.load('./datasets/Abalone.csv')
    
    # plot Y label distribution
    print("Plotting abalone label distribution")
    aba.plot_label_chart('Aba_Label_Distribution', 'Cluster', 'Amount', aba.Y, save=True, path=args.save_path)

    # constant setting
    visualType = True if(args.visual == "save") else False
    X = aba.X
    Y = aba.Y


    if args.algorithm == 'all':
        # predict
        km_sklearn = aba.do_sklearn_Kmeans()
        km_myownkm = aba.do_my_Kmeans(args)
        km_myownkmedoids = aba.do_my_Kmedoids(args, initCentroid=km_myownkm.initCentroid)
        km_myownkmedians = aba.do_my_Kmedians(args, initCentroid=km_myownkm.initCentroid)
        km_myownkkmeans = aba.do_my_KKmeans(args, initCentroid=km_myownkm.initCentroid, kMethod=args.kernel_method)



    

        # visulize result
        print("Saving result images...")
        aba.plot_result("aba-ground_truth", "length", "diameter", "weight.sh", X[:, 0], X[:, 1], X[:, -1],
                            Y, save=visualType, path=args.save_path)
        aba.plot_result("aba-sklearn_kmeans", "length", "diameter", "weight.sh", X[:, 0], X[:, 1], X[:, -1],
                            km_sklearn.labels_, save=visualType, path=args.save_path)
        aba.plot_result("aba-my_kmeans", "length", "diameter", "weight.sh", X[:, 0], X[:, 1], X[:, -1],
                            km_myownkm.labels_, save=visualType, path=args.save_path)
        aba.plot_result("aba-my_kmedoids", "length", "diameter", "weight.sh", X[:, 0], X[:, 1], X[:, -1],
                            km_myownkmedoids.labels_, save=visualType, path=args.save_path)
        aba.plot_result("aba-my_kmedians", "length", "diameter", "weight.sh", X[:, 0], X[:, 1], X[:, -1],
                            km_myownkmedians.labels_, save=visualType, path=args.save_path)
        aba.plot_result("aba-my_kernel-kmeans", "length", "diameter", "weight.sh", X[:, 0], X[:, 1], X[:, -1],
                            km_myownkkmeans.labels_, save=visualType, path=args.save_path)


    else:
        funcdict = {    'kmeans': aba.do_my_Kmeans, 'kmedoids': aba.do_my_Kmedoids,
                        'kmedians': aba.do_my_Kmedians, 'kernel-kmeans': aba.do_my_KKmeans    }
        km_mykm = funcdict[args.algorithm] (args) if args.algorithm != 'kernel-kmeans' else funcdict[args.algorithm] (args, kMethod=args.kernel_method) 

        print("Saving result images...")
        aba.plot_result('aba-my_'+args.algorithm, "length", "diameter", "weight.sh", X[:, 0], X[:, 1], X[:, -1],
                            km_mykm.labels_, save=visualType, path=args.save_path)

        






def main():
    
    # get arguments 
    args = argument_parsing()
    
    if(args.dataset == "iris"):
        do_iris(args)
    elif(args.dataset == "abalone"):
        do_abalone(args)
    else:
        print("Invalid dataset name.")


if __name__ == "__main__":
    main()
