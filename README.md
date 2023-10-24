<small>
**Create Configuration file with following data on corresponding lines:**
  
- **Directory path for ImageNet dataset**
- **Model checkpoint file**
- **Observations file**
- **Continue training data file** 
- **Initial learning rate** 
- **Learning rate decay step size** 
- **Learning rate decay gamma** 
- **Batch size** 
- **Optimizer state file for resumption** 

**The following two files will be created:**

- **Observations file (for every epoch):**
  - Epoch number
  - Learning rate
  - Train loss
  - Train Top1 accuracy
  - Train Top5 accuracy
  - Val loss
  - Val Top1 accuracy
  - Val Top5 accuracy

- **Continue_training file:**
  - Last epoch completed
  - Learning rate for the next epoch
  - Total training hours
  - Total training minutes
  - Total training seconds

**ImageNet directory should contain:**

- train
- test
- val
- LOC_synset_mapping.txt

</small>
