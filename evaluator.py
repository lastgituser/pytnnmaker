#! /usr/bin/python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as tdata
from torchvision import datasets, models, transforms
from PIL import Image
import copy
import time

categories = {
    'img': [
        'trash',
        'compost',
        'recyclable',
    ]
}

numcats = len(categories[categories.keys()[0]])

preprocess = transforms.Compose([
    transforms.Resize((256, 256), Image.ANTIALIAS),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
data_transforms = {
    'train': preprocess,
    'val': preprocess,
}
root_dirs = {
    'train': './' + categories.keys()[0] + '_collective4',
    'val': './' + categories.keys()[0] + ''
}
stages = ['train', 'val']
phase = stages[1]
imgdatasets = {x: datasets.ImageFolder(root_dirs[x], data_transforms[x])
                for x in stages}
imgdataloaders = {x: tdata.DataLoader(imgdatasets[x], batch_size=1, shuffle=True, 
        num_workers=4)
        for x in stages}
dataset_sizes = {x: len(imgdatasets[x])
                for x in stages}
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 9, 3)
        self.conv2 = nn.Conv2d(9, 25, 3)
        self.fc1 = nn.Linear(900, 100)
        self.fc2 = nn.Linear(100, 40)
        self.fc3 = nn.Linear(40, len(imgdatasets['train'].classes))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 900)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#net = Net()
net = models.resnet34(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, numcats)
net.eval()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.1)

try:
    net.load_state_dict(torch.load('model_state_dict_coll4.pkl'))
except:
    pass
running_corrects = 0
running_loss = 0
for i, data in enumerate(imgdataloaders[phase], 0):
    inputs, labels = data
    optimizer.zero_grad()
    with torch.set_grad_enabled(phase == 'train'):
        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
    running_corrects += torch.sum(preds == labels.data)
    running_loss += loss.item() * inputs.size(0)
epoch_acc = running_corrects.double() / dataset_sizes[phase]
epoch_loss = running_loss / dataset_sizes[phase]
print("Acc: %.4f Loss: %.4f" % (epoch_acc, epoch_loss))
