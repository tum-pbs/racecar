# racecar Training

This is the source code repository for our paper
"Data-driven Regularization via Racecar Training for Generalizing Neural Networks",
stay tuned...

![racecar Training teaser](resources/racecar-teaser.jpg)

## Abstract:

We propose a novel training approach for improving the generalization in neural networks.  We show that in contrast to regular constraints for orthogonality, our approach represents a {\em data-dependent} orthogonality constraint, and is closely related to singular value decompositions of the weight matrices.  We also show how our formulation is easy to realize in practical network architectures via a reverse pass, which aims for reconstructing the full sequence of internal states of the network.  Despite being a surprisingly simple change, we demonstrate that this forward-backward training approach, which we refer to as "racecar" training, leads to significantly more generic features being extracted from a given data set.  Networks trained with our approach show more balanced mutual information between input and output throughout all layers, yield improved explainability and, exhibit improved performance for a variety of tasks and task transfers.

Pre-print: <https://arxiv.org/pdf/2007.00024.pdf>

Lab Website: <https://ge.in.tum.de/publications/2020-xie-racecar/>

