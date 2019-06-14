K-means midterm project
===
###### tags: `nsysu` `homework` `big data analysis`



This is a midterm project of big data analysis class in NSYSU. 


The project implement the k-means, k-medoids, k-medain, and kernel k-means algorithms, with random or k-means++ centroid selection.


The testing dataset is iris flower and abalone. 



## Requirement
- python3
- python3 package
    1. sklearn
    2. numpy
    3. pandas
    4. matplotlib



## Usage 
just simply run the command
```
$ python3 main.py
```
:::warning
Note:
    於 Kernel K - means 階段偶有錯誤發生，重新執行即可
:::
and it will run program on iris dataset with 3 clusters and max iteration 300, the arguments will output at the begining.

```
===== Arguments ======
algorithm :  all
cluster :  3
init_method :  random
iter :  300
dataset :  iris
file :  None
visual :  save
save_path :  ./result_img/
kernel_method :  RBF
converge_ratio :  0.7
at_least_iter :  15
======================
```
Btw, the output image will saved at result_img directory.




If you want to run with abalone dataset, just add `--dataset=abalone arguments`
```
$ python3 main.py --dataset=abalone
```


To get other argument usage, you can run with:
```
$ python3 main.py --help
```
to get help message. 



For instance, run with abalone dataset, max iteration 100,  with only kernel k-means algorithm, and polynomial kernel function, the execute command is:
```
$ python3 main.py --dataset=abalone --iter=100 --algorithm=kernel-kmeans --kernel-method=POLY
```



## Author
- B043040003 范真瑋
- B043040020 張哲魁
- B043040039 周家池
- B043040044 吳俊忻




