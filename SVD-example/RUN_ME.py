# ----------------------------------------------------------------------------
#
# Data-driven Regularization via Racecar Training for Generalizing Neural Networks
# Copyright 2020 You Xie, Nils Thuerey
#
# This program is free software, distributed under the terms of the
# Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
# Main script to run full set of SVD tests
#
# ----------------------------------------------------------------------------
import os, shutil

# choose GPU to run on
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# cleanup, remove directories from previous run
paths = ["peak_Ort", "peak_Ort_SVD_0000_0049", "peak_RR", "peak_RR_SVD_0000_0049", "peak_Std", "peak_Std_SVD_0000_0049", "MNIST_Ort", "MNIST_Ort_SVD_0000_0009", "MNIST_RR", "MNIST_RR_SVD_0000_0009", "MNIST_Std", "MNIST_Std_SVD_0000_0009" ]
for path in paths:
    shutil.rmtree(path,ignore_errors=True)

# generate MNIST models and data
os.system('python mnist_training.py --training 1 --testdir MNIST_RR --testnum -1 --modelnum -1 --batchsize 1 --epochnum 1000 --learningrate 0.0001 --maxsave 20 --orthofactor 0.0  --rrfactor 0.00001 --runfile %s'%(__file__))
os.system('python mnist_training.py --training 1 --testdir MNIST_Ort --testnum -1 --modelnum -1 --batchsize 2 --epochnum 1000 --learningrate 0.0001 --maxsave 20 --orthofactor 0.1 --rrfactor 0.0 --runfile %s'%(__file__))
os.system('python mnist_training.py --training 1 --testdir MNIST_Std --testnum -1 --modelnum -1  --batchsize 2 --epochnum 1000 --learningrate 0.0001 --maxsave 20 --orthofactor 0.0 --rrfactor 0.0 --runfile %s'%(__file__))

# generate "peak" models and data
os.system('python peak_training.py --training 1 --testdir peak_RR --testnum -1 --modelnum -1 --batchsize 100 --epochnum 5000 --learningrate 0.0001 --maxsave 20 --orthofactor 0.0 --rrfactor 0.000001 --runfile %s'%(__file__))
os.system('python peak_training.py --training 1 --testdir peak_Ort --testnum -1 --modelnum -1 --batchsize 100 --epochnum 5000 --learningrate 0.0001 --maxsave 20 --orthofactor 0.1 --rrfactor 0.0 --runfile %s'%(__file__))
os.system('python peak_training.py --training 1 --testdir peak_Std --testnum -1 --modelnum -1 --batchsize 100 --epochnum 5000 --learningrate 0.0001 --maxsave 20 --orthofactor 0.0 --rrfactor 0.0 --runfile %s'%(__file__))

# evaluate SVD
os.system("python kernel_SVD.py --testnum %d --modelnum %d --foldername MNIST_RR"%(0,9))
os.system("python kernel_SVD.py --testnum %d --modelnum %d --foldername MNIST_Ort"%(0,9))
os.system("python kernel_SVD.py --testnum %d --modelnum %d --foldername MNIST_Std"%(0,9))
os.system("python kernel_SVD.py --testnum %d --modelnum %d --foldername peak_RR"%(0,49))
os.system("python kernel_SVD.py --testnum %d --modelnum %d --foldername peak_Ort"%(0,49))
os.system("python kernel_SVD.py --testnum %d --modelnum %d --foldername peak_Std"%(0,49))
