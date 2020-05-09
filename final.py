#! /usr/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as tdata
from torchvision import datasets, models, transforms
from PIL import Image
import copy
from bs4 import BeautifulSoup
from icrawler import Parser
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
import PIL
import os
import time
import re

n_epochs = 1

start_phase = 0

categories = {
    'img': [
        'apple fruit',
        'pear fruit',
        'banana fruit',
    ]
}

logfilename = 'logfile.csv'

save_file = 'model_state_dict.pkl'

class GoogleParser(Parser):

    def parse(self, response):
        soup = BeautifulSoup(
            response.content.decode('utf-8', 'ignore'), 'lxml')
        image_divs = soup.find_all(name='script')
        for div in image_divs:
            txt = str(div)
            if 'AF_initDataCallback' not in txt:
                continue
            if 'ds:0' in txt or 'ds:1' not in txt:
                continue
            uris = re.findall(r'http.*?\.(?:jpg|png|bmp)', txt)
            return [{'file_url': uri} for uri in uris]

def dct(categorieslist):
    logfile = open(logfilename, 'a+')
    logfile.write('iteration,time,accuracy,loss,epoch')
    numcats = 0
    for k, v in categories.items():
        numcats += len(v)
    for k, v in categories.items():
        kfolder = './' + k
        otherfolder = './' + k + '_collective'
        if not os.path.exists(otherfolder):
            os.mkdir(otherfolder)
        if not os.path.exists(kfolder):
            os.mkdir(kfolder)
        for c in v:
            logfile.write(',' + c + ' count')
        logfile.write('\n')
        logfile.close()
        start_time = time.time()
        j = 0
        while True:
            for keywords in v:
                kwfolder = keywords.replace(' ', '_', 10)
                newfolder = kfolder + '/' + kwfolder
                othernewfolder = otherfolder + '/' + kwfolder
                if not os.path.exists(othernewfolder):
                    os.mkdir(othernewfolder)
                if not os.path.exists(newfolder):
                    os.mkdir(newfolder)
                crawler = GoogleImageCrawler(parser_cls=GoogleParser,
			storage={'root_dir': newfolder},
                        downloader_threads=4, feeder_threads=1, 
                        parser_threads=1)
                filters = dict(
                    size='medium',
                    date=((2019 - j // 12 // 28, 12 - ((j // 28) % 12), 
                            28 - (j % 28)), 
                            (2018 - j // 12 // 28, 12 - ((j // 28) % 12), 
                            28 - (j % 28))), 
                )
                crawler.crawl(keyword=keywords,
                    filters=filters, max_num = 1000,
                    file_idx_offset='auto')
                for imgname in os.listdir(newfolder):
                    filename = os.path.join(newfolder, imgname)
                    try:
                        img = Image.open(filename).convert('RGB')
                        img.save(filename, 'JPEG')
                    except:
                        os.remove(filename)
            anyfolderempty = False
            for cat in v:
                kwfolder = cat.replace(' ', '_', 10)
                newfolder = kfolder + '/' + kwfolder
                if len(os.listdir(newfolder)) == 0:
                    anyfolderempty = True
            if anyfolderempty:
                continue
            #net = Net()
            #"""
            net = models.resnet34(pretrained=True)
            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, numcats)
            #"""
            try:
                net.load_state_dict(torch.load(save_file ))
            except:
                pass
            net.eval()

            imgdatasets = {x: datasets.ImageFolder(root_dirs[x], data_transforms[x])
                            for x in stages}
            imgdataloaders = {x: tdata.DataLoader(imgdatasets[x], batch_size=1, 
                    shuffle=True, num_workers=4)
                    for x in stages}
            dataset_sizes = {x: len(imgdatasets[x])
                            for x in stages}
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.1)
            
            for epoch in range(n_epochs):
                running_corrects = 0
                running_loss = 0
                for i, data in enumerate(imgdataloaders['train'], 0):
                    inputs, labels = data
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = net(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    running_corrects += torch.sum(preds == labels.data)
                    running_loss += loss.item() * inputs.size(0)
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                epoch_loss = running_loss / dataset_sizes[phase]
                logfile = open(logfilename, 'a+')
                curr_time = int(time.time() - start_time)
                logfile.write("%d,%dd%dh%02dm%02ds,%.4f,%.4f,%d" %
                        (j,
                        curr_time // (60 * 60 * 24), 
                        (curr_time // (60 * 60)) % 24,
                        (curr_time // 60) % 60, 
                        curr_time % 60,
                        epoch_acc, 
                        epoch_loss, 
                        epoch + 1))
                for c in os.listdir(kfolder):
                    cfolder = os.path.join(kfolder, c)
                    logfile.write(',' + str(len(os.listdir(cfolder))))
                logfile.write('\n')
                logfile.close()
                print("Acc: %.4f Loss: %.4f" % (epoch_acc, epoch_loss))

            torch.save(net.state_dict(), save_file)

            for cat in os.listdir(kfolder):
                catdir = os.path.join(kfolder, cat)
                othercatdir = os.path.join(otherfolder, cat)
                for imgfile in os.listdir(catdir):
                    os.remove(os.path.join(catdir, imgfile))
            j += 1

root = list(categories.items())[0][0]

preprocess = transforms.Compose([
    transforms.Resize((256, 256), Image.ANTIALIAS),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
data_transforms = {
    'train': preprocess,
}
root_dirs = {
    'train': './' + root
}
stages = ['train']
phase = stages[0]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 9, 3)
        self.conv2 = nn.Conv2d(9, 25, 3)
        self.fc1 = nn.Linear(900, 100)
        self.fc2 = nn.Linear(100, 40)
        self.fc3 = nn.Linear(40, len(categories[root]))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 900)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    dct(categories)
