# -*- coding:utf-8 -*-
"""
author: Fred
date: 2024.07.23
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os


class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        if self.train:
            self.data_file_names = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        else:
            self.data_file_names = ['test_batch']

        self.images = []
        self.labels = []
        self.load_data()

    def load_data(self):
        for file_name in self.data_file_names:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, 'rb') as f:
                data = pickle.load(f, encoding='bytes')
            if self.train:
                self.images.append(data[b'data'])
                self.labels.extend(data[b'labels'])
            else:
                self.images = data[b'data']
                self.labels = data[b'labels']

        self.images = np.vstack(self.images).reshape(-1, 3, 32, 32)
        self.images = self.images.transpose((0, 2, 3, 1))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, stride=stride,
                               kernel_size=3, padding=1, bias=False)
        # 使用batch normalization时卷积层不使用偏置
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, stride=1,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        x = self.relu(x)

        return x


class Resnet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()
        # 去掉最大池化层

        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 每个通道特征图变为1x1
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel, kernel_size=1, padding=0, stride=stride),
                nn.BatchNorm2d(channel))

        layers = []
        layers.append(block(self.in_channel, channel, stride=stride, downsample=downsample))
        self.in_channel = channel

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), transforms.RandomHorizontalFlip()
                                      , transforms.RandomCrop(32, padding=4)])
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

root_dir = 'cifar-10-python.tar/cifar-10-python/cifar-10-batches-py'
train_dataset = CIFAR10Dataset(root=root_dir, train=True, transform=train_transform)
test_dataset = CIFAR10Dataset(root=root_dir, train=False, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

resnet_18 = Resnet(BasicBlock, [2, 2, 2, 2])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet_18.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

trainloss = []
testloss = []
trainacc = []
testacc = []

best_acc = 0.0
best_epoch = 0


def train(epoch):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # if batch_idx % 100 == 99:  # 每100个批次打印一次损失
        #     print(f'[{epoch + 1}, {batch_idx + 1}] loss: {running_loss:.3f}')
        #     running_loss = 0.0
    trainloss.append(running_loss / len(train_loader))
    train_acc = 100. * correct / total
    trainacc.append(train_acc)
    print(f'Epoch {epoch} Train Loss: {running_loss:.4f} Train Accuracy: {train_acc:.2f}%')


def test(epoch):
    global best_acc, best_epoch
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            c = (predicted == targets).squeeze()
            for i in range(targets.size(0)):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    testloss.append(test_loss / len(test_loader))
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    test_acc = 100. * correct / total
    if test_acc > best_acc:
        best_acc = test_acc
        best_epoch = epoch+1
    testacc.append(test_acc)
    print(f'Epoch {epoch} Test Loss: {test_loss:.4f} Test Accuracy: {test_acc:.2f}%')
    for i in range(10):
        print(f'Accuracy of {classes[i]} : {100 * class_correct[i] / class_total[i]:.2f} %')


epochs = 50
for epoch in range(epochs):
    train(epoch)
    test(epoch)
print(f'Best Accuracy: {best_acc:.2f}%  Best Epoch {best_epoch}')

epochs = np.arange(1, epochs + 1)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(epochs, trainloss, 'g-', label='Train Loss')
ax1.plot(epochs, testloss, 'g--', label='Test Loss')
ax2.plot(epochs, trainacc, 'b-', label='Train Accuracy')
ax2.plot(epochs, testacc, 'b--', label='Test Accuracy')

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='g')
ax2.set_ylabel('Accuracy (%)', color='b')

ax1.legend(loc='upper left')
ax2.legend(loc='center right')

plt.title('Loss and Accuracy')
plt.show()


