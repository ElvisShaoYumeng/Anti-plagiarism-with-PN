import torch
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import pickle as p
import matplotlib.image as plimg
from PIL import Image
import torch.nn as nn

from Nets import cifar_cnn1

EPOCH = 10
BATCH_SIZE = 10
LR = 0.01
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.CIFAR10(root='../data/cifar10/', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=DOWNLOAD_MNIST, )
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


def load_CIFAR_batch(filename):
    with open(filename, 'rb')as f:
        datadict = p.load(f, encoding='latin1')
        x = datadict['data']
        y = datadict['labels']
        x = x.reshape(10000, 3, 32, 32)
        y = np.array(y)
        return x, y


def load_CIFAR_Lables(filename):
    with open(filename, 'rb')as f:
        lines = [x for x in f.readlines()]
        print(lines)


testx, testy = load_CIFAR_batch("../data/cifar10/cifar-10-batches-py/test_batch")

img_x = torch.from_numpy(testx)[:2000]
img_y = torch.from_numpy(testy)[:2000]

test_x = img_x.type(torch.FloatTensor).cuda() / 255.
test_y = img_y.cuda()


class _LeNet(nn.Module):
    def __init__(self):
        super(_LeNet, self).__init__()  # 输入是28*28*1

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),  # 28*28*16 #32*32*16
            nn.MaxPool2d(kernel_size=2),  # 14*14*16
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            # 14*14*32  #16*16*32
            nn.MaxPool2d(kernel_size=2),  # 7*7*32 #8*8*32
        )
        self.linear1 = nn.Linear(8 * 8 * 32, 120)
        self.linear2 = nn.Linear(120, 120)
        self.linear3 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x=x.view(x.size(0),-1)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        output = self.out(x)
        return output


cnn = cifar_cnn1()
cnn.cuda()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
i = 0
# 训练过程，train_loader加载训练数据
for epoch in range(EPOCH):
    for step, (data, labels) in enumerate(train_loader):
        c_x = data.cuda()
        # c_x=x
        c_y = labels.cuda()
        # c_y=y
        output = cnn(c_x)
        loss = loss_func(output, c_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i = i + 1
        # print(i)
        #########训练到此结束##########
        if step % 50 == 0:
            test_out = cnn(test_x)
            pred_y = torch.max(test_out, 1)[1].cuda().data
            num = 0
            for i in range(test_y.size(0)):
                if test_y[i].float() == pred_y[i].float():
                    num = num + 1
            accuracy = num / test_y.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)

test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].cuda().data
print(pred_y, 'prediction numbe')
print(test_y[:10], 'real number')