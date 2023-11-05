import matplotlib.pyplot as plt
import sys
import os
configuration=open(sys.argv[1],'r')
configuration_data=configuration.read().splitlines()

observations_file_name=configuration_data[2]
observations=open(observations_file_name,'r')
observations_data=observations.read().split('\n\n')
while '' in observations_data:
    observations_data.remove('')
for i in range(len(observations_data)):
    observations_data[i]=observations_data[i].split('\n')
    # print(*observations_data[i],sep='\n')

ob=observations_data
    
epochs=[]
learning_rates=[]
train_loss=[]
train_top1_accuracy=[]
train_top5_accuracy=[]
train_time=[]
val_loss=[]
val_top1_accuracy=[]
val_top5_accuracy=[]
val_time=[]

for i in ob:
    epochs.append(int(i[0]))
    learning_rates.append(float(i[1]))
    train_loss.append(float(i[2]))
    train_top1_accuracy.append(float(i[3]))
    train_top5_accuracy.append(float(i[4]))
    t_time=i[5].split(':')
    t_time=int(t_time[0])*60*60+int(t_time[1])*60+int(t_time[2])
    train_time.append(t_time)
    val_loss.append(float(i[6]))
    val_top1_accuracy.append(float(i[7]))
    val_top5_accuracy.append(float(i[8]))
    v_time=i[9].split(':')
    v_time=int(v_time[0])*60*60+int(v_time[1])*60+int(v_time[2])
    val_time.append(v_time)

observations.close()

graphs=[learning_rates, 
        train_loss, 
        train_top1_accuracy, 
        train_top5_accuracy, 
        train_time, 
        val_loss, 
        val_top1_accuracy, 
        val_top5_accuracy, 
        val_time]
graphs_names=['learning_rates',
              'train_loss', 
              'train_top1_accuracy', 
              'train_top5_accuracy', 
              'train_time', 
              'val_loss', 
              'val_top1_accuracy', 
              'val_top5_accuracy', 
              'val_time']
xlabels=['Epochs']*9
ylabels=['Learning Rate',
         'Train Loss',
         'Train Top1 Accuracy',
         'Train Top5 Accuracy',
         'Train Time in seconds',
         'Validation Loss',
         'Validation Top1 Accuracy',
         'Validation Top5 Accuracy',
         'Validation Time in seconds']
titles=['Learning Rate Across Epochs',
         'Train Loss Across Epochs',
         'Train Top1 Accuracy Across Epochs',
         'Train Top5 Accuracy Across Epochs',
         'Train Time Across Epochs',
         'Validation Loss Across Epochs',
         'Validation Top1 Accuracy Across Epochs',
         'Validation Top5 Accuracy Across Epochs',
         'Validation Time Across Epochs']

x_ticks=list(set(epochs))
x_labels = [f'{i}' for i in x_ticks]

for i in range(3):
    for j in range(3):
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.2,right=0.8,top=0.8,bottom=0.2)
        k=i*3+j
        ax.plot(epochs, graphs[k], color='blue', marker='o', linestyle='-', linewidth=1)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('Epochs')
        ax.set_ylabel(ylabels[k])
        ax.set_title(titles[k])
        ax.grid(True, linestyle='--', alpha=0.6)
        # ax.legend()
        plt.savefig(f"{graphs_names[k]}.jpg",format='jpg')

observations_folder=observations_file_name.split('.')
observations_folder=observations_folder[0]
os.system(f"mkdir -p {observations_folder}")
for i in range(9):
    os.system(f"mv {graphs_names[i]}.jpg {observations_folder}/")
