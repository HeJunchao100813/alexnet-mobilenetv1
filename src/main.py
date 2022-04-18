import time
import skimage.io as io
import os
import FC
import torch
import numpy as np
from numpy import pad
from model.lenet import LeNet5
import torch.nn as nn
from src import Activation

net = LeNet5()

for i, (name, param) in enumerate(net.named_parameters()):
    print(name)
    data = np.load("../filter/" + name + ".npy")
    param.data = torch.from_numpy(data)
print()
net.eval()

dir_name = "../mnist_png/mnist_png/training/9"
files = os.listdir(dir_name)
batch_size = 10

for f in range(0, len(files), batch_size):
    pics = []
    for i in range(batch_size):
        load_from = os.path.join(dir_name, files[f + i])
        image = io.imread(load_from, as_gray=True)
        image = pad(image, ((2, 2), (2, 2)), 'median')
        pic = np.array(image / 255).reshape(1, image.shape[0], -1)
        pics.append(pic)
    pics = np.array(pics)
    inputs = torch.tensor(pics, dtype=torch.float32)

    res = inputs
    for i in range(8):
        res = net.convnet[i](res)

    res = res.data.numpy()

    vector = res.reshape(batch_size, -1)
    vector = FC.FullConnect(vector, np.load('../filter/fc.f6.weight.npy'))
    vector = Activation.ReLU(vector)
    vector = FC.FullConnect(vector, np.load('../filter/fc.f7.weight.npy'))

    print("this number is : ", vector.argmax(axis=1))

    time.sleep(1)
