<small>

**Train model by executing the following command:**

&emsp;python3 resnet.py \<configuration-file\>

**For example** `python3 resnet.py configuration1.txt`

**Make graphs based on observations by executing the following command:**

&emsp;python3 make_graph.py \<configuration-file\>

**For example** `python3 make_graph.py configuration1.txt`

**Check sample configuration file before creating one as explained below**

**Create Configuration file with following data on corresponding lines:**

- **Directory path for ImageNet dataset**
- **Model checkpoint filename including .pth extension**
- **Observations filename including .txt extension**
- **Continue training data filename including .txt extension** 
- **Initial learning rate** 
- **Learning rate decay step size** 
- **Learning rate decay gamma** 
- **Batch size** 
- **Checkpoint state filename including .tar extension for resumption** 

**The following two files will be created while training:**

- **Observations file (for every epoch):**
  - Epoch number
  - Learning rate
  - Train loss
  - Train Top1 accuracy
  - Train Top5 accuracy
  - Train time in hh:mm:ss
  - Validation loss
  - Validation Top1 accuracy
  - Validation Top5 accuracy
  - Validation Time in hh:mm:ss

- **Continue_training file:**
  - Last epoch completed
  - Total training time in hh:mm:ss

**ImageNet directory should contain:**

- folder named 'train' with folders with label names and their corresponding training images inside them.
- folder named 'test' with test images.
- folder named 'val' with folders with label names and their corresponding validation images inside them.
- LOC_synset_mapping.txt

**If you encounter any problems or errors, please email me at 210010034@iitdh.ac.in**

</small>
