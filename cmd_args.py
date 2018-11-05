import argparse
import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--command', default='train', choices=['train'])
parser.add_argument('--data', default='cifar10', choices=['cifar10'])
parser.add_argument('--num-classes', type=int, default=10)
parser.add_argument('--data-augmentation', type=bool, default=False)
parser.add_argument('--val-label-corrupt-prob', type=float, default=0.0)
parser.add_argument('--train-label-corrupt-prob', type=float, default=0.0)

parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--SGD', default='True')
parser.add_argument('--learning-rate', type=float, default=0.1)
parser.add_argument('--lr_decay_step', type=int, default=100)
parser.add_argument('--reset_lr', default='False', help='Reset learning rate in between train and retrain in train_custom.py')

parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-4)

parser.add_argument('--eval-full-trainset', type=bool, default=True,
                    help='Whether to re-evaluate the full train set on a fixed model, or simply ' +
                    'report the running average of training statistics')

parser.add_argument('--arch', default='wide-resnet', choices=['wide-resnet', 'mlp'])

parser.add_argument('--wrn-depth', type=int, default=28)
parser.add_argument('--wrn-widen-factor', type=int, default=1)
parser.add_argument('--wrn-droprate', type=float, default=0.0)

parser.add_argument('--mlp-spec', default='512',
                    help='mlp spec: e.g. 512x128x512 indicates 3 hidden layers')

parser.add_argument('--name', default='', help='Experiment name')
parser.add_argument('--debug', default='True', help='Break training after a few steps.')
parser.add_argument('--save_freq', default=1, type=int, help='Epoch frequency to save model weights')
parser.add_argument('--init_weights', default='', help='Absolute path of weights file to restore.')
parser.add_argument('--init_noise_weight',default=0.05,type=float, help='This weight is the variance of the normal distribution inducing the log-normal distribution for multiplicative noise.')
parser.add_argument('--noise_train',default='False', help='Add multiplicative log-normal noise to the weights, with mean 1 and variable variance that can be trained.')
parser.add_argument('--log-uniform-prior',default='False', help='Log uniform prior on the weights, to be invoked if injecting noise into weights.')
parser.add_argument('--log-uniform-low',type=float,default=-8.0)
parser.add_argument('--log-uniform-hi',type=float,default=1.0)
parser.add_argument('--noise_doubling_freq', type=int, default=100000, help='Frequency with which to double noise weights. Used to explore sensitivity of network as noise increases.') # default is so large as to be intended not to occur

def format_experiment_name(args):
  name = args.name
  if name != '' and not name.endswith('/'):
    name += '_'

  name += args.data + '_'
  name += 'corrupt_(%g_%g)' % (args.train_label_corrupt_prob, args.val_label_corrupt_prob)

  name += args.arch
  if args.arch == 'wide-resnet':
    dropmark = '' if args.wrn_droprate == 0 else ('-dr%g' % args.wrn_droprate)
    name += '{0}-{1}{2}'.format(args.wrn_depth, args.wrn_widen_factor, dropmark)
  elif args.arch == 'mlp':
    name += args.mlp_spec

  args.SGD = True if args.SGD == 'True' else False
  if args.SGD:
    name += '_lr{0}_mmt{1}'.format(args.learning_rate, args.momentum)
  else:
    name += '_lr{0}_adam'.format(args.learning_rate)
  if args.weight_decay > 0:
    name += '_Wd{0}'.format(args.weight_decay)
  else:
    name += '_NoWd'
  if not args.data_augmentation:
    name += '_NoAug'

  timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
  name += '_{0}'.format(timestamp)

  for attr in ['debug', 'reset_lr', 'noise_train', 'log_uniform_prior']:
    setattr(args,attr,True if getattr(args,attr) in ['True','T', True] else False)
  if args.debug:
    name = 'debug/' + name
    args.epochs = 1

  return name


def parse_args(custom_args=None):
  args = parser.parse_args()
  if custom_args:
    for k,v in custom_args.items():
      setattr(args,k,v)
  args.exp_name = format_experiment_name(args)
  return args
