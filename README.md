Create the below mentioned configuration file.<br>
Configuration file:<br>
<ul>
<li>directory path for imagenet</li><br>
<li>file name for model pth file</li><br>
<li>file name for observations</li><br>
<li>file name thats contains data to continue training</li><br>
<li>start learning rate</li><br>
<li>learning rate decay step size</li><br>
<li>learning rate decay gamma</li><br>
<li>batch size</li><br>
<li>file name for optimizer state file for resumption</li><br>
</ul>
<br><br>
The following two files will be created:<br>
Observations file (for every epoch):
<ul>
<li>Epoch number</li><br>
<li>Learning rate</li><br>
<li>Train Loss</li><br>
<li>Train Top1 Accuracy</li><br>
<li>Train Top5 Accuracy</li><br>
<li>Val Loss</li><br>
<li>Val Top1 Accuracy</li><br>
<li>Val Top5 Accuracy</li><br>
</ul>
<br>
Continue_training File:
<ul>
<li>Last epoch completed</li><br>
<li>Learning rate for next epoch</li><br>
<li>Total training hours</li><br>
<li>Total training minutes</li><br>
<li>Total training seconds</li><br>
</ul>
<br>
imagenet directory should contain train,test,val, and LOC_synset_mapping.txt
