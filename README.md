# Configuration file to created in the following way:

# Directory path for ImageNet dataset

# File name for the model checkpoint (.pth file)

# File name for observations

# File name that contains data for continuing training

# Initial learning rate

# Learning rate decay step size

# Learning rate decay gamma

# Batch size

# File name for optimizer state file for resumption

# The following two files will be created:

# Observations file (for every epoch):
# - Epoch number
# - Learning rate
# - Train Loss
# - Train Top1 Accuracy
# - Train Top5 Accuracy
# - Val Loss
# - Val Top1 Accuracy
# - Val Top5 Accuracy

# Continue_training File:
# - Last epoch completed
# - Learning rate for the next epoch
# - Total training hours
# - Total training minutes
# - Total training seconds

# ImageNet directory should contain train, test, val, and LOC_synset_mapping.txt
