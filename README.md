Configuration file:
directory path for imagenet
file name for model pth file
file name for observations
file name thats contains data to continue training
start learning rate
learning rate decay step size
learning rate decay gamma
batch size
file name for optimizer state file for resumption

Observations file (for every epoch):
Epoch number
Learning rate
Train Loss
Train Top1 Accuracy
Train Top5 Accuracy
Val Loss
Val Top1 Accuracy
Val Top5 Accuracy

Continue_training File:
Last epoch completed
Learning rate for next epoch
Total training hours
Total training minutes
Total training seconds

imagenet should contain train,test,val, and LOC_synset_mapping.txt