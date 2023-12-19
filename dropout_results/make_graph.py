image_format="jpg"

import matplotlib.pyplot as plt
import sys
import os

f1=open('1/observations1.txt','r')
f2=open('2/observations1.txt','r')
f3=open('3/observations1.txt','r')
f4=open('4/observations1.txt','r')

d1=f1.read().split('\n\n')
d2=f2.read().split('\n\n')
d3=f3.read().split('\n\n')
d4=f4.read().split('\n\n')
d=[d1,d2,d3,d4]
for i in d:
    while '' in i:
        i.remove('')
    for j in range(len(i)):
        i[j]=i[j].split('\n')
        # print(*observations_data[i],sep='\n')

min_obs=min([len(d1),len(d2),len(d3),len(d4)])
    
epochs=[[],[],[],[]]
learning_rates=[[],[],[],[]]
train_loss=[[],[],[],[]]
train_top1_accuracy=[[],[],[],[]]
train_top5_accuracy=[[],[],[],[]]
train_time=[[],[],[],[]]
val_loss=[[],[],[],[]]
val_top1_accuracy=[[],[],[],[]]
val_top5_accuracy=[[],[],[],[]]
val_time=[[],[],[],[]]

for k,j in enumerate(d):
    for m,i in enumerate(j):
        if m>=min_obs:
            continue
        epochs[k].append(int(i[0]))
        learning_rates[k].append(float(i[1]))
        train_loss[k].append(float(i[2]))
        train_top1_accuracy[k].append(float(i[3]))
        train_top5_accuracy[k].append(float(i[4]))
        t_time=i[5].split(':')
        t_time=int(t_time[0])*60*60+int(t_time[1])*60+int(t_time[2])
        train_time[k].append(t_time)
        val_loss[k].append(float(i[6]))
        val_top1_accuracy[k].append(float(i[7]))
        val_top5_accuracy[k].append(float(i[8]))
        v_time=i[9].split(':')
        v_time=int(v_time[0])*60*60+int(v_time[1])*60+int(v_time[2])
        val_time[k].append(v_time)

f1.close()
f2.close()
f3.close()
f4.close()

graphs=[
        # learning_rates, 
        train_loss, 
        train_top1_accuracy, 
        train_top5_accuracy, 
        # train_time, 
        val_loss, 
        val_top1_accuracy, 
        val_top5_accuracy 
        # ,val_time
        ]
graphs_names=[
            #   'learning_rates',
              'train_loss', 
              'train_top1_accuracy', 
              'train_top5_accuracy', 
            #   'train_time', 
              'val_loss', 
              'val_top1_accuracy', 
              'val_top5_accuracy' 
            #   ,'val_time'
              ]
xlabels=['Epochs']*6
ylabels=[
        # 'Learning Rate',
         'Train Loss',
         'Train Top1 Accuracy',
         'Train Top5 Accuracy',
        #  'Train Time in seconds',
         'Validation Loss',
         'Validation Top1 Accuracy',
         'Validation Top5 Accuracy'
        #  ,'Validation Time in seconds'
        ]
titles=[
        # 'Learning Rate Across Epochs',
         'Train Loss Across Epochs',
         'Train Top1 Accuracy Across Epochs',
         'Train Top5 Accuracy Across Epochs',
        #  'Train Time Across Epochs',
         'Validation Loss Across Epochs',
         'Validation Top1 Accuracy Across Epochs',
         'Validation Top5 Accuracy Across Epochs'
        #  ,'Validation Time Across Epochs'
        ]

x_ticks=[i for i in range(1,len(epochs[0])+1)]
x_ticklabels = [f'{i}' for i in x_ticks]
y_ticks=[i for i in range(0,101,10)]
y_ticklabels = [f'{i}' for i in y_ticks]

line_labels=["0.0","0.1","0.3","0.5"]
line_colors=["blue","red","orange","black"]

for i in range(2):
    for j in range(3):
        fig, ax = plt.subplots(figsize=(50,25))
        plt.subplots_adjust(left=0.12,right=0.88,top=0.88,bottom=0.12)
        k=i*3+j
        for x in range(4):
            #print(len(epochs[0]),len(graphs[k][x]))
            #print(graphs[k][x])
            ax.plot(epochs[0], graphs[k][x], color=line_colors[x], marker='.', linestyle='-', linewidth=1,label=line_labels[x])
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticklabels)
        if graphs[k] in [train_top1_accuracy,train_top5_accuracy,val_top1_accuracy,val_top5_accuracy]:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_ticklabels)
        ax.set_xlabel('Epochs')
        ax.set_ylabel(ylabels[k])
        ax.set_title(titles[k])
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        plt.savefig(f"{graphs_names[k]}.{image_format}",format=image_format,dpi=350)
