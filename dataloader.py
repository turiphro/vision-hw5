import torch
import torchvision
import torchvision.transforms as transforms


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.247, 0.243, 0.261)


class CifarLoader(object):
	"""docstring for CifarLoader"""
	def __init__(self, args):
		super(CifarLoader, self).__init__()
		transform = transforms.Compose(
		    [
		     transforms.RandomHorizontalFlip(),
		     transforms.RandomCrop(32, padding=4),
		     transforms.RandomRotation(5, expand=False),
		     transforms.ToTensor(),
		     transforms.Normalize(mean=MEAN, std=STD)
		     ])

		transform_test = transforms.Compose([
		    transforms.ToTensor(),
		    transforms.Normalize(mean=MEAN, std=STD)
		])

		trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
		                                        download=True, transform=transform)
		self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize,
		                                          shuffle=True, num_workers=2)

		testset = torchvision.datasets.CIFAR10(root='./data', train=False,
		                                       download=True, transform=transform_test) 
		self.testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize,
		                                         shuffle=False, num_workers=2)

		self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
		
