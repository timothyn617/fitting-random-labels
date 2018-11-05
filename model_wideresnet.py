# Wide Resnet model adapted from https://github.com/xternalz/WideResNet-pytorch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
  def __init__(self, in_planes, out_planes, stride, dropRate=0.0, noise_train=True):
    super(BasicBlock, self).__init__()
    self.bn1 = nn.BatchNorm2d(in_planes)
    self.relu1 = nn.ReLU(inplace=True)
    self.conv1 = Conv2dwithNoise(in_planes, out_planes, kernel_size=3, stride=stride,
                            padding=1, bias=False, noise_train=noise_train)
    self.bn2 = nn.BatchNorm2d(out_planes)
    self.relu2 = nn.ReLU(inplace=True)
    self.conv2 = Conv2dwithNoise(out_planes, out_planes, kernel_size=3, stride=1,
                            padding=1, bias=False, noise_train=noise_train)
    self.droprate = dropRate
    self.equalInOut = (in_planes == out_planes)
    self.convShortcut = (not self.equalInOut) and Conv2dwithNoise(in_planes, out_planes, kernel_size=1, stride=stride,
                                                            padding=0, bias=False, noise_train=noise_train) or None

  def forward(self, x):
    if not self.equalInOut:
      x = self.relu1(self.bn1(x))
      out = self.conv1(x)
    else:
      out = self.conv1(self.relu1(self.bn1(x)))

    if self.droprate > 0:
      out = F.dropout(out, p=self.droprate, training=self.training)
    out = self.conv2(self.relu2(self.bn2(out)))
    if not self.equalInOut:
      return torch.add(self.convShortcut(x), out) # convolutional res-connection (to force equality of input and output channel dimension)
    else:
      return torch.add(x, out)

class Conv2dwithNoise(torch.nn.Conv2d):
  def __init__(self, in_planes, out_planes, kernel_size, stride, padding, bias=False, noise_train=True):
    super(Conv2dwithNoise, self).__init__(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    self.noise_train = noise_train
    if noise_train:
      self.noise_alpha = torch.nn.parameter.Parameter(torch.zeros(self.weight.size()), requires_grad=noise_train)

  def get_log_normal(self):
    return get_log_normal(self)

  def forward(self, input):
    weight = self.weight
    if self.noise_train:
      weight = self.get_log_normal() * self.weight
    return F.conv2d(input, weight, self.bias, self.stride,
             self.padding, self.dilation, self.groups)

class LinearwithNoise(torch.nn.Linear):

  def __init__(self, in_features, out_features, bias=True, noise_train=True):
    super(LinearwithNoise, self).__init__(in_features, out_features, bias=bias)
    self.noise_train = noise_train
    if noise_train:
      self.noise_alpha = torch.nn.parameter.Parameter(torch.zeros(self.weight.size()), requires_grad=noise_train)

  def get_log_normal(self):
    return get_log_normal(self)

  def forward(self, input):
    weight = self.weight
    if self.noise_train:
      weight = self.get_log_normal() * self.weight
    return F.linear(input, weight, self.bias)

def get_log_normal(module):
  alpha = torch.abs(module.noise_alpha)
  if torch.cuda.is_available():
    eps = torch.cuda.FloatTensor(module.weight.size()).normal_()
  else:
    eps = torch.FloatTensor(module.weight.size()).normal_()
  sample = -alpha / 2 + torch.pow(alpha,0.5) * eps
  return torch.exp(sample)

def get_log_uniform(size, lo=-8.0, hi=1.0):
  # log uniform prior in the range [-lo,-hi] and [lo,hi]
  positive_weights = torch.exp(torch.distributions.uniform.Uniform(torch.tensor(lo), torch.tensor(hi)).rsample(size))
  random_signs = torch.sign(torch.Tensor(size).random_(0, 2) - 0.5)
  return random_signs*positive_weights

class NetworkBlock(nn.Module):
  def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, noise_train=True):
    super(NetworkBlock, self).__init__()
    self.layer = self._make_layer(
        block, in_planes, out_planes, nb_layers, stride, dropRate, noise_train)

  def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, noise_train):
    layers = []
    for i in range(nb_layers):
        layers.append(block(i == 0 and in_planes or out_planes, # i = 0: in_planes; i > 0: out_planes
                            out_planes, i == 0 and stride or 1, dropRate, noise_train=noise_train)) # i = 0: stride; i > 0: 1
    return nn.Sequential(*layers)

  def forward(self, x):
    return self.layer(x)


class WideResNet(nn.Module):
  def __init__(self, depth, num_classes, widen_factor=1, drop_rate=0.0, noise_train=True):
    super(WideResNet, self).__init__()

    nChannels = [16, 16 * widen_factor,
                  32 * widen_factor, 64 * widen_factor]
    assert((depth - 4) % 6 == 0)
    n = (depth - 4) // 6
    block = BasicBlock
    # 1st conv before any network block
    self.conv1 = Conv2dwithNoise(3, nChannels[0], kernel_size=3, stride=1,
                            padding=1, bias=False, noise_train=noise_train)
    # 1st block
    self.block1 = NetworkBlock(
        n, nChannels[0], nChannels[1], block, 1, drop_rate,noise_train)
    # 2nd block
    self.block2 = NetworkBlock(
        n, nChannels[1], nChannels[2], block, 2, drop_rate,noise_train)
    # 3rd block
    self.block3 = NetworkBlock(
        n, nChannels[2], nChannels[3], block, 2, drop_rate,noise_train)
    # global average pooling and classifier
    self.bn1 = nn.BatchNorm2d(nChannels[3])
    self.relu = nn.ReLU(inplace=True)
    self.fc = LinearwithNoise(nChannels[3],num_classes,noise_train=noise_train)
    self.nChannels = nChannels[3]

  def log_uniform_init(self, lo, hi):
    for m in self.modules():
      if isinstance(m, Conv2dwithNoise):
        m.weight.data = get_log_uniform(m.weight.size(), lo, hi)
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, LinearwithNoise):
        m.weight.data = get_log_uniform(m.weight.size(), lo, hi)
        m.bias.data = get_log_uniform(m.bias.size(), lo, hi)

  def var_scaling_init(self, init_scale):
    for m in self.modules():
      if isinstance(m, Conv2dwithNoise):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, init_scale * math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, LinearwithNoise):
        m.bias.data.zero_()
        size = m.weight.size()
        fan_out = size[0] # number of rows
        fan_in = size[1] # number of columns
        variance = math.sqrt(2.0/(fan_in + fan_out))
        m.weight.data.normal_(0.0, init_scale * variance)

  def init_noise_weights(self, init_noise_weight):
    for m in self.modules():
      if isinstance(m, Conv2dwithNoise):
        m.noise_alpha.data.fill_(init_noise_weight)
      elif isinstance(m, LinearwithNoise):
        m.noise_alpha.data.fill_(init_noise_weight)

  def forward(self, x):
    out = self.forward_repr(x)
    return self.fc(out)

  def forward_repr(self, x): # outputs final feature layer
    out = self.conv1(x) # (B, 16, 32, 32)
    out = self.block1(out) # (B, 16, 32, 32)
    out = self.block2(out) # (B, 32, 16, 16)
    out = self.block3(out) # (B, 64, 8, 8)
    out = self.relu(self.bn1(out))
    out = F.avg_pool2d(out, 8) # (B, 64, 1, 1)
    out = out.view(-1, self.nChannels) # (B, 64)
    return out

