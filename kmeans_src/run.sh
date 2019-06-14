#!/bin/bash

mkdir -p result/iris_k3/
mkdir -p result/iris_k3/random_RBF/
mkdir -p result/iris_k3/kpp_POLY/

python3 main.py --dataset=iris --cluster=3 --save-path=result/iris_k3/random_RBF/ --kernel-method=RBF > result/iris_k3/random_RBF/iris_k3_random.log
python3 main.py --dataset=iris --cluster=3 --save-path=result/iris_k3/kpp_POLY/ --kernel-method=POLY  --init-method=k-means++ > result/iris_k3/kpp_POLY/iris_k3_POLY.log





for i in 3 4 7 28; 
do 
mkdir -p result/abalone_k"$i"/
mkdir -p result/abalone_k"$i"/random_RBF/
mkdir -p result/abalone_k"$i"/kpp_POLY/
python3 main.py --dataset=abalone --cluster="$i" --save-path=result/abalone_k"$i"/random_RBF/ --kernel-method=RBF > result/abalone_k"$i"/random_RBF/abalone_k"$i"_random.log
python3 main.py --dataset=abalone --cluster="$i" --save-path=result/abalone_k"$i"/kpp_POLY/ --kernel-method=POLY --init-method=k-means++ > result/abalone_k"$i"/kpp_POLY/abalone_k"$i"_random.log
done
