import torch
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import numpy as np
import pickle




def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


def t():
    os.system("python ./model/main.py")

def r(testimgs,model,tf):

	#Selecting a random row from testimg pickle file
    testimg=testimgs[np.random.choice(testimgs.shape[0])]
    
    img = testimg.reshape(3, 32, 32).transpose(1,2,0).astype("uint8")
    imgtens=tf.transform_test()
    testimage=imgtens(img)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    testimage = testimage.view(1, 3, 32,32)
    plt.imsave(('.//flask_api//static//image'+'//res.png'),img)
    res=model(testimage)
    return classes[int(res.max(1)[1])]
    
def u(model,tf,path):
    img=plt.imread(path)
    imgtens=tf.transform_test()
    testimage=imgtens(img)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    testimage = testimage.view(1, 3, 32,32)
    res=model(testimage)
    return classes[int(res.max(1)[1])]
		#plt.figure(figsize=(2,2))
		#plt.imshow(img)









