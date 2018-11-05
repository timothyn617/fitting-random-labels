"""
cifar-10 dataset, with support for random labels
"""
import numpy as np

import torch
import torchvision.datasets as datasets


class CIFAR10RandomLabels(datasets.CIFAR10):
  """CIFAR10 dataset, with support for randomly corrupt labels.

  Params
  ------
  corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
  num_classes: int
    Default 10. The number of classes in the dataset.
  """
  def __init__(self, corrupt_prob=0.0, num_classes=10, **kwargs):
    super(CIFAR10RandomLabels, self).__init__(**kwargs)
    #self.n_classes = num_classes
    if corrupt_prob > 0:
      corrupt_labels(self, corrupt_prob, num_classes)

def corrupt_labels(self, corrupt_prob, num_classes): # self is a dataset
  labels = np.array(self.train_labels if self.train else self.test_labels)
  np.random.seed(12345)
  mask = np.random.rand(len(labels)) <= corrupt_prob
  rnd_labels = np.random.choice(num_classes, mask.sum())
  labels[mask] = rnd_labels
  # we need to explicitly cast the labels from npy.int64 to
  # builtin int type, otherwise pytorch will fail...
  labels = [int(x) for x in labels]

  if self.train:
    self.train_labels = labels
  else:
    self.test_labels = labels

def get_mixed_dataset(args):
  train_ds = datasets.CIFAR10(train=True)
  if args.train_label_corrupt_prob > 0:
    corrupt_labels(train_ds, args.train_label_corrupt_prob,10)
  val_ds = datasets.CIFAR10(train=False)
  if args.val_label_corrupt_prob > 0:
    corrupt_labels(val_ds, args.val_label_corrupt_prob,10)
  concat_ds = datasets.ConcatDataset([train_ds,val_ds])
  return concat_ds
