

import torch 
import torch.nn as nn 
import torch.optim as optim 
from torchvision import datasets, transforms 
import matplotlib.pyplot as plt 
# plt.use('Agg')

# transform method, a class that you can pass in with different transform method 
# later it can take in the data(images or audio) and transform the data into tensor 
# class "transform" -> method 1: ToTensor() -> specify it so that transform module can use
transform = transforms.ToTensor() 

# define the dataset 
mnist_data = datasets.MNIST(root='./data', train=True,
                            download=True, transform=transform) 

# dataloader 
data_loader = torch.utils.data.DataLoader(dataset=mnist_data,
                                          batch_size=64,
                                          shuffle=True)

# inspect -> tensor range: tensor(0.) tensor(1.)
# later in the output layer, we need to make sure that all data fits in this range
images, labels = next(iter(data_loader))
print(torch.min(images), torch.max(images))


# see what's inside dataloader 
first_sample = mnist_data[0]
print(first_sample)
