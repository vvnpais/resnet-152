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
    def __init__(self,block,layers,image_channels,num_classes):
        super(ResNet,self).__init__()
        self.conv1=nn.Conv2d(image_channels,64,kernel_size=7,stride=2,padding=3)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.in_channels=64

        # Resnet layers
        self.layer1=self._make_layer(block,layers[0],out_channels=64,stride=1)
        self.layer2=self._make_layer(block,layers[1],out_channels=128,stride=2)
        self.layer3=self._make_layer(block,layers[2],out_channels=256,stride=2)
        self.layer4=self._make_layer(block,layers[3],out_channels=512,stride=2)
        
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
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
    
def ResNet50(img_channels=3,num_classes=1000):
    return ResNet(block,[3,4,6,3],img_channels,num_classes)
def ResNet101(img_channels=3,num_classes=1000):
    return ResNet(block,[3,4,23,3],img_channels,num_classes)
def ResNet152(img_channels=3,num_classes=1000):
    return ResNet(block,[3,8,36,3],img_channels,num_classes)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=ResNet152(num_classes=1000).to(device)

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

print(data_directory,pth_file,OBSERVATIONS,CONTINUE_TRAINING,start_learning_rate,step_size,gamma,batch_size,checkpoint_state_file)
print(type(data_directory),type(pth_file),type(OBSERVATIONS),type(CONTINUE_TRAINING),type(start_learning_rate),type(step_size),type(gamma),type(batch_size),type(checkpoint_state_file))

os.system(f'touch {CONTINUE_TRAINING}')
continue_training_file=open(CONTINUE_TRAINING,'r')
continue_training=continue_training_file.read().splitlines()
# print(continue_training_data)
if continue_training!=[]:
    continue_training[1]=continue_training[1].split(":")
    last_completed_epoch_number=int(continue_training[0])
    training_hours=int(continue_training[1][0])
    training_minutes=int(continue_training[1][1])
    training_seconds=int(continue_training[1][2])
    continue_training_boolean=True
else:
    last_completed_epoch_number=0
    training_hours=0
    training_minutes=0
    training_seconds=0
    continue_training_boolean=False
continue_training_file.close()

##########################################################################################################

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
    ])
}
sets=['train','val']
image_datasets={ x :datasets.ImageFolder(os.path.join(data_directory,x),data_transforms[x]) for x in ['train','val']}
num_work=4
dataloaders={}
dataloaders['train']=torch.utils.data.DataLoader(image_datasets['train'],batch_size=batch_size,shuffle=True,num_workers=min(num_work,os.cpu_count()))
dataloaders['val']=torch.utils.data.DataLoader(image_datasets['val'],batch_size=batch_size,shuffle=False,num_workers=min(num_work,os.cpu_count()))
dataset_sizes={x:len(image_datasets[x]) for x in ['train','val']}
class_names=image_datasets['train'].classes
# print(class_names)

##########################################################################################################

def synset():
    synset=open(data_directory+'LOC_synset_mapping.txt')
    syn=synset.read().split('\n')
    while '' in syn:
        syn.remove('')
    for i in range(len(syn)):
        # print(syn[i])
        ind=syn[i].index(' ')
        syn[i]=[syn[i][:ind],syn[i][ind+1:]]
    syn={i+1:syn[i] for i in range(len(syn))}
    return syn


def checkDataLoader():
    syn=synset()
    samples,labels=next(iter(dataloaders['train']))
    print(samples.dtype,labels.dtype,sys.getsizeof(samples[0].storage()),samples.shape)
    print(samples[0][0],labels[0])
    for i in range(9):
        print(syn[labels[i].item()])
    plt.subplot(3,3,1+i)
    grid_img=torchvision.utils.make_grid(samples[i],nrow=3)
    plt.imshow(grid_img.permute(1,2,0))
    plt.show()
    exit()

def totalParameters():
    for i in model.parameters():
        print(i.dtype)
        break
    a=0
    for param in model.parameters():
        t=param.view(-1)
        a+=t.shape[0]
    print(a)
    print(f'{a//1000000},{(a//1000)%1000:03d},{a%1000:03d}')
    exit()
    
def parameterShapes():
    for param in model.parameters():
        print(param.shape)
    exit()

def cudaUsageStats():
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    
# uncomment below line to check if images are correct according to their labels. this function will give proper class names in output
# checkDataLoader()

#uncomment below line to view parameter dataype and total number of parameters
# totalParameters()

#uncomment below line to view shapes of parameter-tensors
# parameterShapes()

##########################################################################################################

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=start_learning_rate)
scheduler=lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)

if continue_training_boolean:
    model.load_state_dict(torch.load(pth_file))
    checkpoint_state=torch.load(checkpoint_state_file)
    optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint_state['scheduler_state_dict'])


for param in model.parameters():
    param.requires_grad=True

#uncomment below line to check for val accuracy
#sets=['val']

alpha=0.5*10**(-5)
while True:
    train_loss=0
    train_top1_accuracy=0
    train_top5_accuracy=0
    train_time_h=0
    train_time_m=0
    train_time_s=0
    val_loss=0
    val_top1_accuracy=0
    val_top5_accuracy=0
    val_time_h=0
    val_time_m=0
    val_time_s=0
    epoch_learning_rate=0
    
    epoch_start=time.time()
    print(f'Epoch No: {last_completed_epoch_number+1}')
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
    for phase in sets:
        phase_start_time=epoch_start if phase=='train' else time.time()
        if phase=='train':
            model.train() #set model to training mode
        else:
            model.eval() #set model to evaluation mode
            
        running_loss=torch.tensor(0.0).to(device)
        running_corrects=torch.tensor(0.0).to(device)
        top5k=torch.tensor(0.0).to(device)

        for inputs,labels in dataloaders[phase]:
            inputs=inputs.to(device)
            labels=labels.to(device)
        
            with torch.set_grad_enabled(phase=='train'):
                outputs=model(inputs)
                # cudaUsageStats()
                _,preds=torch.max(outputs,1)
                sum_w2=0
                for param in model.parameters():
                    sum_w2+=torch.sum(torch.square(param))
                L2=alpha*sum_w2
                loss=criterion(outputs,labels)+L2
                
                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
            now=time.time()
            phaseval=phase.capitalize()
            print(f'Phase:{phaseval} Loss: {loss.item():03.9f}, Runtime: {(now-phase_start_time)//3600:02.0f}h, {((now-phase_start_time)//60)%60:02.0f}m, {(now-phase_start_time)%60:06.3f}s L2:{L2.item()}',end='\r')
            
            running_loss+=loss.item()*inputs.size(0)
            running_corrects+=torch.sum(preds==labels.data)
            for i,j in zip(outputs,labels):
                v=torch.sort(i,descending=True)
                if j in v[1][:5]:
                    top5k+=1
        epoch_loss=running_loss/dataset_sizes[phase]
        epoch_acc=running_corrects.double()/dataset_sizes[phase]
        epoch_top5k_acc=top5k/dataset_sizes[phase]
        if phase=='train':
            epoch_learning_rate=optimizer.param_groups[0]['lr']
            scheduler.step()
        
        phaseval=phase.capitalize()    
        print(f'{phaseval} Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.4f} % Top5k: {epoch_top5k_acc*100:.4f} %                  ')
        
        now=time.time()
        training_hours+=int((now-phase_start_time)//3600)
        training_minutes+=int(((now-phase_start_time)//60)%60)
        training_seconds+=int((now-phase_start_time)%60)
        
        if phase=='train':
            train_loss=epoch_loss
            train_top1_accuracy=epoch_acc*100
            train_top5_accuracy=epoch_top5k_acc*100
            train_time_h=int((now-phase_start_time)//3600)
            train_time_m=int(((now-phase_start_time)//60)%60)
            train_time_s=int((now-phase_start_time)%60)
        elif phase=='val':
            val_loss=epoch_loss
            val_top1_accuracy=epoch_acc*100
            val_top5_accuracy=epoch_top5k_acc*100
            val_time_h=int((now-phase_start_time)//3600)
            val_time_m=int(((now-phase_start_time)//60)%60)
            val_time_s=int((now-phase_start_time)%60)
                
        print(f'Epoch time:{(now-phase_start_time)//3600:02.0f}h, {((now-phase_start_time)//60)%60:02.0f}m, {(now-phase_start_time)%60:02.3f}s')
        while training_seconds>=60:
            training_seconds-=60
            training_minutes+=1
        while training_minutes>=60:
            training_minutes-=60
            training_hours+=1
        
        if phase=='val' and sets!=['val']:
            torch.save(model.state_dict(),pth_file)
            last_completed_epoch_number+=1
            observations_file=open(OBSERVATIONS,'a')
            observations_file.write(f'{last_completed_epoch_number}\n')
            observations_file.write(f'{epoch_learning_rate}\n')
            observations_file.write(f'{train_loss}\n')
            observations_file.write(f'{train_top1_accuracy}\n')
            observations_file.write(f'{train_top5_accuracy}\n')
            observations_file.write(f'{train_time_h}:{train_time_m:02d}:{train_time_s:02d}\n')
            observations_file.write(f'{val_loss}\n')
            observations_file.write(f'{val_top1_accuracy}\n')
            observations_file.write(f'{val_top5_accuracy}\n')
            observations_file.write(f'{val_time_h}:{val_time_m:02d}:{val_time_s:02d}\n\n')
            observations_file.close()
            continue_training_file=open(CONTINUE_TRAINING,'w')
            continue_training_file.write(f'{last_completed_epoch_number}\n')
            continue_training_file.write(f'{training_hours}:{training_minutes:02d}:{training_seconds:02d}\n')
            continue_training_file.close()
            torch.save({
                'optimizer_state_dict':optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict()
                },checkpoint_state_file)
            
    print()
