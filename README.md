This repo is an enrichment of this [repo](https://github.com/pluskid/fitting-random-labels), which provides code to train models that memorize (random) labels on CIFAR-10 and is based on the paper 

> [1] Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals. *Understanding deep learning requires rethinking generalization*. International Conference on Learning Representations (ICLR), 2017. [[arXiv:1611.03530](https://arxiv.org/abs/1611.03530)].

It's main intention is to provide additional training options and tensorboard summaries that allow further probing of memorizing networks. In particular, we allow for

### 1) Multiplicative noise in weights:

This implements ideas of 

> [2] Diederik P. Kingma, Tim Salimans, Max Welling. *Variational Dropout and the Local Reparameterization Trick.* Advances in Neural Information Processing Systems 28 (NIPS 2015). [[arXiv:1506.02557]]

> [3] Alessandro Achille, Stefano Soatto. *Emergence of Invariance and Disentanglement in Deep Representations.* Journal of Machine Learning Research 18 (2018). [[arXiv:1706.01350]]

We introduce noise into our neural network via a multiplicative log-normal noise. This has a variational Bayesian formulation [2] and provides a way of measuring the information-theoretic content stored by the weights of a neural network [3]. We corroborate some of the observations in [3] by showing that neural networks which memorize random labels are more sensitive to noise than those which memorize true labels. This is obtained by freezing the model, letting the noise variance increase, and plotting the train and test accuracy. The below plots illlustrate the results as follows. We let the noise variance increase every 3 epochs, which thus decreases the information (upper bound) stored by the neural net (left plot). The performance degrades most quickly for random label memorizing networks versus the true label memorizing networks (each have 3 separate runs, the former group deteriorates around epoch 15, the latter around epoch 25) (center plot). Note that the networks that memorize random labels have validation accuracy consistent with random guessing (10%) (right plot).

<p align="center">
<img src="https://github.com/timothyn617/fitting-random-labels/blob/noisy_weights/tb1.png">
</p>

### 2) Catastrophic Forgetting

In train_custom.py, we allow for two runs of training, the first of which trains on both the training and validation set (each with their own label corruption percent), and then retraining on just the training portion. The main intention is to investigate what happens when a net is trained to memorize both random train labels and true validation labels and then trains on only the former - to what extent is performance on the latter going to be affected? We found that catastrophic forgetting / instability can occurr: 

<p align="center">
<img src="https://github.com/timothyn617/fitting-random-labels/blob/noisy_weights/tb2.png">
</p>

The blue curve is the initial run on both the train and validation set, for which it achieves (near) 100% accuracy. The red curve is the net continuing to train but with the validation set removed. There is a sudden degradation of performance on both the train and validation set near epoch 90. The performance on the train set recovers, while the validation does not.

### 3) Accuracy Oscillation

We plot the intersection and union of correct labels for consecutive epochs (as percentages of the dataset). This allows us to understand how the accuracy per epoch is distributed among examples over the course of training. We see below that early in training when accuracy is low and moderate, the correct examples fluctuate, as indicated by the union being significantly larger than the intersection. These are three different runs on random train labels:

<p align="center">
<img src="https://github.com/timothyn617/fitting-random-labels/blob/noisy_weights/tb3.png">
</p>

