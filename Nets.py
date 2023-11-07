import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation




class mnist_cnn(nn.Module):
    def __init__(self):
        super().__init__()
        # 128x28
        self.conv1=nn.Conv2d(1,10,5)         # 10, 24x24
        self.conv2=nn.Conv2d(10, 20,3)       #128, 10x10
        self.fc1=nn.Linear(20*10*10, 500)
        self.fc2=nn.Linear(500, 10)
    def forward(self, x):
        in_size=x.size(0)		# in_size 为 batch_size（一个batch中的Sample数）
        # 卷积层 -> relu -> 最大池化
        out = self.conv1(x)     # 24
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)  # 12
        #卷积层 -> relu -> 多行变一行 -> 全连接层 -> relu -> 全连接层 -> sigmoid
        out = self.conv2(out)  # 10
        out = F.relu(out)
        out = out.view(in_size, -1)     # view()函数作用是将一个多行的Tensor,拼接成一行。
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        # softmax
        out = F.log_softmax(out, dim=1)
        # 返回值 out
        return out


'''class cifar_cnn(nn.Module):
    def __init__(self):
        super(cifar_cnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x'''

class cifar_cnn(nn.Module):
    def __init__(self):
        super(cifar_cnn,self).__init__()
      	# 输入shape 3*32*32
        self.conv1 = nn.Conv2d(3,64,3,padding=1)        # 64*32*32
        self.conv2 = nn.Conv2d(64,64,3,padding=1)       # 64*32*32
        self.pool1 = nn.MaxPool2d(2, 2)                 # 64*16*16
        self.bn1 = nn.BatchNorm2d(64)                   # 64*16*16
        self.relu1 = nn.ReLU()                          # 64*16*16

        self.conv3 = nn.Conv2d(64,128,3,padding=1)      # 128*16*16
        self.conv4 = nn.Conv2d(128, 128, 3,padding=1)   # 128*16*16
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)      # 128*9*9
        self.bn2 = nn.BatchNorm2d(128)                  # 128*9*9
        self.relu2 = nn.ReLU()                          # 128*9*9

        self.conv5 = nn.Conv2d(128,128, 3,padding=1)    # 128*9*9
        self.conv6 = nn.Conv2d(128, 128, 3,padding=1)   # 128*9*9
        self.conv7 = nn.Conv2d(128, 128, 1,padding=1)   # 128*11*11
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)      # 128*6*6
        self.bn3 = nn.BatchNorm2d(128)                  # 128*6*6
        self.relu3 = nn.ReLU()                          # 128*6*6

        self.conv8 = nn.Conv2d(128, 256, 3,padding=1)   # 256*6*6
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)  # 256*6*6
        self.conv10 = nn.Conv2d(256, 256, 1, padding=1) # 256*8*8
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)      # 256*5*5
        self.bn4 = nn.BatchNorm2d(256)                  # 256*5*5
        self.relu4 = nn.ReLU()                          # 256*5*5

        self.conv11 = nn.Conv2d(256, 512, 3, padding=1) # 512*5*5
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1) # 512*5*5
        self.conv13 = nn.Conv2d(512, 512, 1, padding=1) # 512*7*7
        self.pool5 = nn.MaxPool2d(2, 2, padding=1)      # 512*4*4
        self.bn5 = nn.BatchNorm2d(512)                  # 512*4*4
        self.relu5 = nn.ReLU()                          # 512*4*4

        self.fc14 = nn.Linear(512*4*4,1024)             # 1*1024
        self.drop1 = nn.Dropout2d()                     # 1*1024
        self.fc15 = nn.Linear(1024,1024)                # 1*1024
        self.drop2 = nn.Dropout2d()                     # 1*1024
        self.fc16 = nn.Linear(1024,10)                  # 1*10

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)


        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        # print(" x shape ",x.size())
        x = x.view(-1,512*4*4)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = self.fc16(x)

        return x

# ----------------------------------------------------------------------------------------------------------------------

import torch.nn.functional as F


class cifar_cnn1(nn.Module):
    def __init__(self):
        super(cifar_cnn1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.globalavgpool = nn.AvgPool2d(8, 8)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout10 = nn.Dropout(0.1)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn1(F.relu(self.conv2(x)))
        x = self.maxpool(x)
        x = self.dropout10(x)
        x = self.bn2(F.relu(self.conv3(x)))
        x = self.bn2(F.relu(self.conv4(x)))
        x = self.avgpool(x)
        x = self.dropout10(x)
        x = self.bn3(F.relu(self.conv5(x)))
        x = self.bn3(F.relu(self.conv6(x)))
        x = self.globalavgpool(x)
        x = self.dropout50(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# 构建网络
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16        # 64, 3, 32, 32
        self.conv = conv3x3(3, 16)       # 64, 16, 32, 32
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(True)
        #self.layer1 = self.make_layer(block, 16, layers[0])    # 64, 16, 32, 32
        #self.layer2 = self.make_layer(block, 32, layers[0], 2)   # 64, 32, 16, 16
        #self.layer3 = self.make_layer(block, 64, layers[1], 2)    # 64, 64, 8, 8
        #self.avg_pool = nn.AvgPool2d(8)        # 64, 64, 1,
        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)    # 64, 16, 32, 32
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)   # 64, 32, 16, 16
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)    # 64, 64, 8, 8
        #self.layer4 = self.make_layer(block, 512, layers[3], stride=2)    # 64, 64, 8, 8
        self.fc1 = nn.Linear(512*32, num_classes)

    def make_layer(self, block, out_channles, blocks, stride=1):
        downsample = None
        if out_channles != self.in_channels or stride != 1:
            downsample = nn.Sequential(conv3x3(self.in_channels, out_channles, stride=stride), nn.BatchNorm2d(out_channles))
        layers = []
        layers.append(block(self.in_channels, out_channles, stride, downsample))
        self.in_channels = out_channles
        for i in range(1, blocks):
            layers.append(block(out_channles, out_channles))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out



cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()

class SVHN_nn(nn.Module):
    def __init__(self):
        super(SVHN_nn, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.globalavgpool = nn.AvgPool2d(8, 8)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout10 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(64*64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout50(x)
        x = self.fc2(x)
        return x

class SVHN_nn1(nn.Module):
    def __init__(self):
        super(SVHN_nn1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.globalavgpool = nn.AvgPool2d(8, 8)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout10 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(64*64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, network):
        #network = input_data(shape=[None, 32, 32, 3])
        network = conv_2d(network, 32, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = conv_2d(network, 64, 3, activation='relu')
        network = conv_2d(network, 64, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = fully_connected(network, 512, activation='relu')
        network = dropout(network, 0.5)
        network = fully_connected(network, 10, activation='softmax')
        network = regression(network, optimizer='adam',
                             loss='categorical_crossentropy',learning_rate=0.001)
        return network

class Adult_nn(nn.Module):
    def __init__(self):
        super(Adult_nn, self).__init__()
        print("Adult_NN: MLP is created")
        self.l1 = nn.Linear(10,32)
        self.l2 = nn.Linear(32,8)
        self.l3 = nn.Linear(8,2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class Adult_dnn(nn.Module):
    def __init__(self):
        super(Adult_dnn, self).__init__()
        print("Adult_DNN: MLP is created")
        self.l1 = nn.Linear(10, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 32)
        self.l4 = nn.Linear(32, 8)
        self.l5 = nn.Linear(8, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = self.l5(x)
        return x