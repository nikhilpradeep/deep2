# loading all necessay packages for visualization.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
import seaborn as sn  # for heatmaps
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os


def eda():
    transform = transforms.ToTensor()
    
    
    train_data = datasets.CIFAR10(root='./src/data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./src/data', train=False, download=True, transform=transform)

	
    torch.manual_seed(101)  # just in case anyone wants to reproduce the results

    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

    class_names = ['plane', '  car', ' bird', '  cat', ' deer', '  dog', ' frog', 'horse', ' ship', 'truck']


    np.set_printoptions(formatter=dict(int=lambda x: f'{x:5}')) # to widen the printed array

	# Grab the first batch of 10 images
    for images,labels in train_loader: 
        break

	# Print the labels
    classes = np.array([class_names[i] for i in labels])
    sh1 = len(train_data.targets)
    sh2 = len(test_data.targets)
    print('Label:', labels.numpy())
    print('Class: ', *np.array([class_names[i] for i in labels]))

	# Print the images
    im = make_grid(images, nrow=5)  # the default nrow is 8
    plt.figure(figsize=(10,4))
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))

    if os.path.exists('.//flask_api//static//image')==False:
        os.mkdir('.//flask_api//static//image')

    if os.path.exists('.//flask_api//static//image'+'//cifar10.png')==False:
        plt.imsave(('.//flask_api//static//image'+'//cifar10.png'),np.transpose(im.numpy(), (1, 2, 0)))
    
    results = [sh1,sh2,class_names,labels.numpy(),classes]
    return results