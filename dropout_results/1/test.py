import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import os
import copy
import sys



########################################## RESNET DEFINITION #############################################

class block(nn.Module):
    def __init__(self,in_channels,out_channels,identity_downsample=None,stride=1):
        super(block,self).__init__()
        self.expansion=4
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.conv3=nn.Conv2d(out_channels,out_channels*self.expansion,kernel_size=1,stride=1,padding=0)
        self.bn3=nn.BatchNorm2d(out_channels*self.expansion)
        self.relu=nn.ReLU()
        self.identity_downsample=identity_downsample

    def forward(self,x):
        identity=x

        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu(x)
        x=self.conv3(x)
        x=self.bn3(x)

        if self.identity_downsample is not None:
            identity=self.identity_downsample(identity)

        x+=identity
        x=self.relu(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self,block,layers,image_channels,num_classes,dropout_probability):
        super(ResNet,self).__init__()
        self.conv1=nn.Conv2d(image_channels,64,kernel_size=7,stride=2,padding=3)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.in_channels=64
        self.dropout_probability=dropout_probability

        # Resnet layers
        self.layer1=self._make_layer(block,layers[0],out_channels=64,stride=1)
        self.layer2=self._make_layer(block,layers[1],out_channels=128,stride=2)
        self.layer3=self._make_layer(block,layers[2],out_channels=256,stride=2)
        self.layer4=self._make_layer(block,layers[3],out_channels=512,stride=2)
        
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.dropout=nn.Dropout(p=dropout_probability)
        self.fc=nn.Linear(512*4,num_classes)

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        x=self.avgpool(x)
        x=x.reshape(x.shape[0],-1)
        x=self.dropout(x)
        x=self.fc(x)
        return x

    def _make_layer(self,block,num_residual_blocks,out_channels,stride):
        identity_downsample=None
        layers=[]

        if stride!=1 or self.in_channels!=out_channels*4:
            identity_downsample=nn.Sequential(nn.Conv2d(self.in_channels,out_channels*4,kernel_size=1,stride=stride),nn.BatchNorm2d(out_channels*4))

        layers.append(block(self.in_channels,out_channels,identity_downsample,stride))
        self.in_channels=out_channels*4 #64*4

        for i in range(num_residual_blocks-1):
            layers.append(block(self.in_channels,out_channels))

        return nn.Sequential(*layers)
    

##########################################################################################################

CONFIGURATION=sys.argv[1]
configuration_file=open(CONFIGURATION,'r')
configuration=configuration_file.read().splitlines()
data_directory=configuration[0]
pth_file=configuration[1]
OBSERVATIONS=configuration[2]
CONTINUE_TRAINING=configuration[3]
start_learning_rate=float(configuration[4])
step_size=int(configuration[5])
gamma=float(configuration[6])
batch_size=int(configuration[7])
checkpoint_state_file=configuration[8]
required_epochs=int(configuration[9])
dropout_probability=float(configuration[10])

print(data_directory,pth_file,OBSERVATIONS,CONTINUE_TRAINING,start_learning_rate,step_size,gamma,batch_size,checkpoint_state_file,required_epochs,dropout_probability,sep="  ")
print(type(data_directory),type(pth_file),type(OBSERVATIONS),type(CONTINUE_TRAINING),type(start_learning_rate),type(step_size),type(gamma),type(batch_size),type(checkpoint_state_file),type(required_epochs),type(dropout_probability),sep="  ")


def ResNet50(img_channels=3,num_classes=1000):
    return ResNet(block,[3,4,6,3],img_channels,num_classes,dropout_probability)
def ResNet101(img_channels=3,num_classes=1000):
    return ResNet(block,[3,4,23,3],img_channels,num_classes,dropout_probability)
def ResNet152(img_channels=3,num_classes=1000):
    return ResNet(block,[3,8,36,3],img_channels,num_classes,dropout_probability)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=ResNet152(num_classes=1000).to(device)


mean=np.array([0.485 , 0.456 , 0.406])
std=np.array([0.229 , 0.224 , 0.225])


data_transforms={
    'train':transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ,transforms.Normalize(mean,std)
    ]),
    'val':transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ]),
    'test':transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
}


FILE="resnet.pth"

continue_training=True
if continue_training:
    model.load_state_dict(torch.load(FILE))
    
for param in model.parameters():
    param.requires_grad=False
    
    
# for i in range(1,100001):
#     img=cv2.imread(f'/home/vivian/imagenet_site/test/test/ILSVRC2012_test_00{i:06d}.JPEG',cv2.IMREAD_COLOR)
    

test_dir='/home/vivian/vv/imagenet_site/'
image_datasets={ x :datasets.ImageFolder(os.path.join(test_dir,x),data_transforms[x]) for x in ['test']}
    
dataloader=torch.utils.data.DataLoader(image_datasets['test'],batch_size=1,shuffle=False,num_workers=min(4,os.cpu_count()))

model.eval()

synset=open('../kagglefiles/files/LOC_synset_mapping.txt')
syn=synset.read().split('\n')
while '' in syn:
    syn.remove('')
for i in range(len(syn)):
    # print(syn[i])
    ind=syn[i].index(' ')
    syn[i]=[syn[i][:ind],syn[i][ind+1:]]
syn={i+1:syn[i] for i in range(len(syn))}

OUTPUT=open("test_output.txt","w")

i=0
for inputs,labels in dataloader:
    print(i)
    i+=1
    inputs=inputs.to(device)
    labels=labels.to(device)
    
    with torch.set_grad_enabled(False):
        outputs=model(inputs)
        _,pred=torch.max(outputs,1)
        # print(pred)
    OUTPUT.write(f"ILSVRC2012_test_{i:08d}.JPEG : {pred.item():03d} : {i:06d} : {syn[pred.item()+1]}\n")


























