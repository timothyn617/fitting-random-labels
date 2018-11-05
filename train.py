from __future__ import print_function

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim
import datetime
import glob
import shutil

from cifar10_data import CIFAR10RandomLabels

import cmd_args
import model_mlp, model_wideresnet
from tensorboardX import SummaryWriter


def get_data_loaders(args, shuffle_train=True,normalize_data=True):
  if args.data == 'cifar10':
    if normalize_data:
      normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                       std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    else:
      normalize = transforms.RandomRotation(0) # hack

    if args.data_augmentation:
      transform_train = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize,
          ])
    else:
      transform_train = transforms.Compose([
          transforms.ToTensor(),
          normalize,
          ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        CIFAR10RandomLabels(root='./data', train=True, download=False,
                            transform=transform_train, num_classes=args.num_classes,
                            corrupt_prob=args.train_label_corrupt_prob),
        batch_size=args.batch_size, shuffle=shuffle_train, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        CIFAR10RandomLabels(root='./data', train=False,
                            transform=transform_test, num_classes=args.num_classes,
                            corrupt_prob=args.val_label_corrupt_prob),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader
  else:
    raise Exception('Unsupported dataset: {0}'.format(args.data))


def get_model(args):
  # create model
  if args.arch == 'wide-resnet':
    model = model_wideresnet.WideResNet(args.wrn_depth, args.num_classes,
                                        args.wrn_widen_factor,
                                        drop_rate=args.wrn_droprate,
                                        noise_train=args.noise_train,
                                        )
    if args.log_uniform_prior:
      model.log_uniform_init(args.log_uniform_low, args.log_uniform_hi)
    else:
      model.var_scaling_init(init_scale=1.0)
    if args.noise_train:
      model.init_noise_weights(args.init_noise_weight)
  elif args.arch == 'mlp':
    if args.noise_train:
      raise NotImplementedError
    n_units = [int(x) for x in args.mlp_spec.split('x')] # hidden dims
    n_units.append(args.num_classes)  # output dim
    n_units.insert(0, 32*32*3)        # input dim
    model = model_mlp.MLP(n_units)

  # for training on multiple GPUs.
  # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
  # model = torch.nn.DataParallel(model).cuda()
  if args.init_weights:
    load_weights(model,args.init_weights)
    #if args.noise_train:
    for name, param in model.named_parameters():
      if 'noise_alpha' in name:
        print('initializing noise_alpha', name)
        param.data.fill_(1e-4)

  model = model.cuda()

  return model

def _get_optimizer(model, args):
  if not args.noise_train:
    parameters = []
    for name, param in model.named_parameters():
      if 'noise_alpha' not in name: parameters.append(param)
  else:
    parameters = model.parameters()
  if args.SGD:
    return torch.optim.SGD(parameters, args.learning_rate,
                  momentum=args.momentum, weight_decay=args.weight_decay)
  else:
    return torch.optim.Adam(parameters, args.learning_rate,
                  weight_decay=args.weight_decay)

def train_model(args, model, train_loader, val_loader=None,
                start_epoch=None, epochs=None):
  cudnn.benchmark = True

  # define loss function (criterion) and pptimizer
  criterion = nn.CrossEntropyLoss().cuda()
  if hasattr(model,'optimizer'):
    optimizer = model.optimizer
  else:
    model.optimizer = optimizer = _get_optimizer(model,args)
  start_epoch = start_epoch or 0
  epochs = (epochs or args.epochs)
  tensorboard_dir = os.path.join(_get_save_dir(args), 'tensorboard')
  writer = SummaryWriter(tensorboard_dir)

  prev_train_pred, prev_val_pred = None, None
  for epoch in range(start_epoch, epochs):
    adjust_learning_rate(optimizer, epoch, args)

    # train for one epoch
    tr_loss, tr_prec1 = train_epoch(train_loader, model, criterion, optimizer, epoch, args, writer)
    if epoch > 0 and epoch % args.noise_doubling_freq == 0:
      print('doubling noise')
      for name, param in model.named_parameters():
        if 'noise_alpha' in name:
          param.data.fill_(2*param.data.view(-1)[0])
    # evaluate on validation set
    if val_loader is not None:
      val_loss, val_prec1, all_labels, all_pred = validate_epoch(val_loader, model, criterion,args)
      writer.add_scalar('data/val/loss_avg_per_epoch', val_loss, epoch)
      writer.add_scalar('data/val/top1_avg_per_epoch', val_prec1, epoch)
      int_pct, union_pct = _get_stats(all_labels, all_pred, all_prev_pred=prev_val_pred)
      writer.add_scalar('data/val/cons_int_pct', int_pct, epoch)
      writer.add_scalar('data/val/cons_union_pct', union_pct, epoch)
      prev_val_pred = all_pred
    else:
      val_loss, val_prec1 = None, None
    # evaluate on train set
    if args.eval_full_trainset:
      nonshuffled_train_loader, _ = get_data_loaders(args, shuffle_train=False)
      tr_loss, tr_prec1, all_labels, all_pred = validate_epoch(nonshuffled_train_loader, model, criterion,args)
      writer.add_scalar('data/train/loss_avg_per_epoch', tr_loss, epoch)
      writer.add_scalar('data/train/top1_avg_per_epoch', tr_prec1, epoch)
      int_pct, union_pct = _get_stats(all_labels, all_pred, all_prev_pred=prev_train_pred)
      writer.add_scalar('data/train/cons_int_pct', int_pct, epoch)
      writer.add_scalar('data/train/cons_union_pct', union_pct, epoch)
      prev_train_pred = all_pred
    if val_loss is not None:
      logging.info('%03d: Acc-tr: %6.2f, Acc-val: %6.2f, L-tr: %6.4f, L-val: %6.4f',
                 epoch, tr_prec1, val_prec1, tr_loss, val_loss)
    else:
      logging.info('%03d: Acc-tr: %6.2f, L-tr: %6.4f',
                   epoch, tr_prec1, tr_loss)
    _save_step(model, args, epoch)
  return model

def _get_save_dir(args):
  return os.getcwd() + '/saved_models/%s' % args.exp_name


def _save_step(model, args, epoch, force=False):
  save_dir = _get_save_dir(args)
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  if force or epoch % args.save_freq == 0:
    # print('saving %s' % save_dir + '/epoch%s.pytorch' % str(epoch).zfill(3))
    torch.save(model.state_dict(), save_dir + '/epoch%s.pytorch' % str(epoch).zfill(3))

def train_epoch(train_loader, model, criterion, optimizer, epoch, args, writer):
  """Train for one epoch on the training set"""
  batch_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()

  # switch to train mode
  model.train()
  dataset_size = len(train_loader.dataset) # 50k train images for cifar
  writer.add_scalar('data/train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
  for i, (input, target) in enumerate(train_loader):
    total_iteration_no = i + epoch* int(np.ceil(dataset_size / args.batch_size))
    target = target.cuda(async=True)
    input = input.cuda()
    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)

    # compute output
    output = model(input_var)
    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1, _ = accuracy(output.data, target, topk=(1,))
    losses.update(loss.data.item(), input.size(0))
    top1.update(prec1[0].item(), input.size(0))  # k = 1, so prec1 is a list of 1 item
    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i % 10 == 0):
      weights_norm = get_weights_norm(model)
      writer.add_scalar('data/train/weights_norm', weights_norm, total_iteration_no)
      writer.add_scalar('data/train/log_loss_per_batch', np.log(losses.val), total_iteration_no)
      grad_norms = map(lambda t: torch.norm(t), [param.grad for param in model.parameters() if param.grad is not None])
      grad_norm = torch.norm(torch.stack(list(grad_norms)))
      writer.add_scalar('data/train/log_gradient_norm_per_batch', torch.log(grad_norm), total_iteration_no)
      if args.debug:
        break
  if args.noise_train:
    alphas_list = get_alphas(model)
    alphas = torch.abs(torch.cat([x.view(-1) for x in alphas_list]))
    information = -0.5 * torch.mean(torch.log(torch.clamp(torch.abs(alphas),1e-10,None)))
    writer.add_scalar('data/train/information_UB', information, epoch)
  return losses.avg, top1.avg


def get_weights_norm(model):
  D = model.state_dict()
  weights = [D[k] for k in D if ('weight' in k or 'bias' in k)]
  return torch.norm(torch.stack(list(map(lambda t: torch.norm(t), weights)))).item()

def get_alphas(model):
  D = model.state_dict()
  alphas = [D[k] for k in D if 'noise_alpha' in k]
  return alphas

def _get_args():
  from collections import namedtuple
  args_dict = vars(cmd_args.parser)
  args_actions = args_dict['_actions']
  args_dict = {}
  for action in args_actions:
    args_dict[action.dest] = action.default
  args_dict = {k: v for k, v in args_dict.items() if not k.startswith('_')}
  Argstuple = namedtuple('Argstuple', args_dict.keys())
  args = Argstuple(**args_dict)
  return args




def load_weights(model,path):
  # loads weights from path into model via the shared set of keys
  state_dict = torch.load(path)
  own_state = model.state_dict()
  for name, param in state_dict.items():
    if name not in own_state:
      print('Warning: saved model has missing key in model to be loaded: %s' % name)
      continue
    param = param.data
    own_state[name].copy_(param)
  not_loaded_keys = set(own_state.keys()) - set(state_dict.keys())
  print('Keys that were not loaded:', not_loaded_keys)
  return model

def validate_epoch(data_loader, model, criterion, args):
  """Perform validation on data given by data_loader. Dataset_type is used to determ"""
  batch_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()

  # switch to evaluate mode
  model.eval()
  pred_list = []
  labels_list = []
  for i, (input, target) in enumerate(data_loader,1):
    target = target.cuda(async=True)
    input = input.cuda()
    input_var = torch.autograd.Variable(input, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)

    # compute output
    output = model(input_var)
    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1, pred = accuracy(output.data, target, topk=(1,))
    losses.update(loss.data[0], input.size(0))
    top1.update(prec1[0].item(), input.size(0))
    pred_list.append(pred)
    labels_list.append(target)
    if args.debug:
      break
  all_pred = torch.cat(pred_list)
  all_labels = torch.cat(labels_list)
  return losses.avg, top1.avg, all_labels, all_pred

def _get_stats(all_labels, all_pred, all_prev_pred):
  if all_prev_pred is None:
    int_pct, union_pct = 0.0, 0.0
  else:
    prev_acc = torch.where(all_prev_pred == all_labels, torch.ones_like(all_pred), torch.zeros_like(all_pred))
    curr_acc = torch.where(all_pred == all_labels, torch.ones_like(all_pred), torch.zeros_like(all_pred))
    intersection = torch.min(torch.stack([prev_acc, curr_acc]),dim=0)[0]
    union = torch.max(torch.stack([prev_acc,curr_acc]),dim=0)[0]
    int_pct, union_pct = torch.mean(intersection.type(torch.float32)).item(), torch.mean(union.type(torch.float32)).item()
  return int_pct, union_pct


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
  """Sets the learning rate to the initial LR decayed by 10 after args.lr_decay_step steps"""
  lr = args.learning_rate * (0.1 ** min((epoch // args.lr_decay_step),1))
  for param_group in optimizer.param_groups:
      param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  assert len(topk) == 1
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True) #pred shape = (B,k)
  pred = pred.t() #(k,B)
  correct = pred.eq(target.view(1, -1).expand_as(pred)) #(k,B)

  res = []
  for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0)
      res.append(correct_k.mul_(100.0 / batch_size)) # each element of res is a tensor
  return res, pred.view(-1)


def setup_logging(args):
  save_dir = _get_save_dir(args)
  if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
  log_fn = os.path.join(save_dir, "LOG.{0}.txt".format(datetime.date.today().strftime("%y%m%d")))
  logging.basicConfig(filename=log_fn, filemode='w', level=logging.DEBUG)
  # also log into console
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  logging.getLogger('').addHandler(console)
  print('Logging into %s...' % save_dir)

def setup_save(model, args):
  _save_step(model, args, epoch=0, force=True)  # save initialization, for reproducibility
  save_dir = _get_save_dir(args)
  with open(save_dir + '/args.txt', 'w+') as f:
    f.write(str(vars(args)))
  os.makedirs(save_dir + '/code')
  scripts = glob.glob(os.getcwd() + '/*.py')
  for script in scripts:
    shutil.copy(script, save_dir + '/code/' + os.path.basename(script))  # save state of code, for reproducibility


def main(custom_args=None):
  args = cmd_args.parse_args(custom_args)
  setup_logging(args)

  model = get_model(args)
  setup_save(model, args)
  logging.info('Number of parameters: %d', sum([p.data.nelement() for p in model.parameters()]))
  train_loader, val_loader = get_data_loaders(args, shuffle_train=True)
  train_model(args, model, train_loader, val_loader)


if __name__ == '__main__':
  main()