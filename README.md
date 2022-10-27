# Reviving Autoencoder Pretraining

This is the source code repository for our paper
"Reviving Autoencoder Pretraining" (published in Neural Computing and Applications Journal).
This repository contains example code to train the MNIST and peak models mentioned in our paper. We also provide an example to analyze the weights via SVD to verify differences among racecar trained model (RR), standard model (Std) and models trained with orthogonal constraints (Ort). I.e., the code here can be used to reproduce Fig. 3 in our paper.


![racecar Training teaser](resources/racecar-teaser.jpg)

## Abstract:

The pressing need for pretraining algorithms has been diminished by numerous advances in terms of regularization, architectures, and optimizers. Despite this trend, we re-visit the classic idea of unsupervised autoencoder pretraining and propose a modified variant that relies on a full reverse pass trained in conjunction with a given training task. This yields networks that are {\em as-invertible-as-possible}, and share mutual information across all constrained layers. We additionally establish links between singular value decomposition and pretraining and show how it can be leveraged for gaining insights about the learned structures. Most importantly, we demonstrate that our approach yields an improved performance for a wide variety of relevant learning and transfer tasks ranging from fully connected networks over residual neural networks to generative adversarial networks. Our results demonstrate that unsupervised pretraining has not lost its practical relevance in today’s deep learning environment.

Paper: <https://link.springer.com/article/10.1007/s00521-022-07892-0>

Lab Website: <https://ge.in.tum.de/publications/2020-xie-racecar/>

## Prerequisites

* [tensorflow](https://www.tensorflow.org/install) >= 1.14
* [numpy](https://numpy.org/install/)  >= 1.18.1

## Files

* Folder "MNIST/" and "peak/" contain the data sets used for MNIST and peak tests, respectively.
* "mnist_training.py" and "peak_training.py" are used to run the MNIST and peak training.
* After training, "kernel_SVD.py" can be used to compute the SVD of the weights in the first layer of the trained models.

## Running the tests
You can run the whole chain by executing
```
python RUN_ME.py
```

![An example evolution of the reverse pass for Std, Ort and RR.](resources/svd-output.gif)

This will train all three models: Std, Ort and RR for both tests, and will create the following outputs:
* the directories "MNIST|peak_RR|Ort|Std/test_0000/" with trained models, and training information (training loss in "trainloss.txt", the elapsed time for 100 epochs in "elapsedtime.txt", testing accuracy in "testaccuracy.txt")
* "MNIST|peak_RR|Ort|Std/test_0000/backwardtest_img/" contain re-generated images $d_{1}^{'}$, i.e. they execute inverted network via the backward path, over the course of the full training process. These outputs are shown in the GIF above for three examples (with Std, Ort, RR f.l.t.r. in each).
* "MNIST|peak_RR|Ort|Std_SVD_0000_0049/" contain images of the SVD of the first layer's weight matrix of the trained model. By default, we only output 10 images from the right singular vectors. More right/left singular vectors can be generated by modifying line 38-43 in "kernel_SVD.py".
