import matplotlib.pyplot as plt
import numpy as np
import argparse
import models
import torch
import torch.optim as optim


def log(str):
	print(str)


def model_class(class_name):
    return getattr(models, class_name)


def argParser():
	parser = argparse.ArgumentParser(description='PyTorch Homework')
	parser.add_argument('--lr', default=0.1, type=float)
	parser.add_argument('--batchSize', default=64, type=int)
	parser.add_argument('--epochs', default=1, type=int)
	parser.add_argument('--model', type=model_class)
	parser.add_argument('--device', default='cpu') # cpu, cuda, cuda:1
	return parser.parse_args()

