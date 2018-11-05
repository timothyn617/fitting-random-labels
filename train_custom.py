from __future__ import print_function

import logging
import torch
import torchvision.transforms as transforms
import torch.optim
import cifar10_data
import torchvision.datasets as datasets
import cmd_args
import train


def get_mixed_loader(args, shuffle_train=True, normalize_data=True):
	if args.data == 'cifar10':
		if normalize_data:
			normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
											 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
		else:
			normalize = transforms.RandomRotation(0)  # hack

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

		train_ds = datasets.CIFAR10(root='./data', train=True, transform=transform_train)
		if args.train_label_corrupt_prob > 0:
			cifar10_data.corrupt_labels(train_ds, args.train_label_corrupt_prob, 10)
		val_ds = datasets.CIFAR10(root='./data', train=False, transform=transform_test)
		if args.val_label_corrupt_prob > 0:
			cifar10_data.corrupt_labels(val_ds, args.val_label_corrupt_prob, 10)
		concat_ds = torch.utils.data.dataset.ConcatDataset([train_ds, val_ds])
		dataloader = torch.utils.data.DataLoader(
			concat_ds,
			batch_size=args.batch_size, shuffle=shuffle_train, **kwargs)

		return dataloader
	else:
		raise Exception('Unsupported dataset: {0}'.format(args.data))

def _train(args, model):
	train_mixed_loader = get_mixed_loader(args)
	model = train.train_model(args, model, train_mixed_loader, val_loader=None) # train on both training and validation data with their own corruption levels
	args.val_label_corrupt_prob = 0.0
	print('retraining...')
	if not args.reset_lr:
		print('using same learning rate...')
		args.learning_rate = model.optimizer.param_groups[0]['lr']
		args.lr_decay_step = 10000000 # no decay step
	train_loader, val_loader = train.get_data_loaders(args)
	args.exp_name += '_retrain'
	train.train_model(args, model, train_loader, val_loader) # train on just training data

def main(custom_args=None):
	args = cmd_args.parse_args(custom_args)
	train.setup_logging(args)
	model = train.get_model(args)
	train.setup_save(model,args)
	logging.info('Number of parameters: %d', sum([p.data.nelement() for p in model.parameters()]))
	_train(args, model)

if __name__ == '__main__':
	main()