import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import os
import time
import itertools
import csv

#Read the raw ET file
dataset=pd.read_csv(r'C:\Users\Christos\Desktop\Memory guided attention in cluttered scenes v.3\Experimental Tasks\Task1\ET_Data\L_17_2021_Mar_30_1821.tsv', sep='\t')
scene_folder='C:\\Users\\Christos\\Desktop\\Memory guided attention in cluttered scenes v.3\\Experimental Tasks\\Task1\\Scenes\\'




#Calculate sampling rate for the entire dataset
def sampling_rate(datafile):
    datafile['Sampling_Rate']=datafile['TIME']-datafile['TIME'].shift()
    datafile['Cumulative_SR']=datafile['Sampling_Rate'].cumsum()
    #sr_graph=datafile.plot.line(x='TIME', y='Sampling_Rate', figsize=(50,3))
    
    return datafile['Sampling_Rate'].describe()
    
sampling_rate(dataset) #example

learning_duration=dataset.Cumulative_SR.max()/60 #example





# Keep only recordings during scene viewing (main trials)
def keep_events(datafile):
    global df_main
    df_main=datafile[datafile['USER'].astype(str).str.startswith('S')]
    
    return df_main

keep_events(dataset) #example




#Keep only trials for which a valid recording exists
def keep_valid_recs(datafile):
    global df_main_valid
    df_main_valid=datafile[datafile['LPOGV']==1]
    
    return df_main_valid

keep_valid_recs(df_main) #example




#Convert gaze data from gp3 coordinates to psychopy coordinates
def convert_gaze_coords(datafile):
    df_main_valid['BPOGX']=(datafile['BPOGX']-0.5)*1920   
    df_main_valid['BPOGY']=-1*(datafile['BPOGY']-0.5)*1080
    df_main_valid['FPOGX']=(datafile['FPOGX']-0.5)*1920       
    df_main_valid['FPOGY']=-1*(datafile['FPOGY']-0.5)*1080
    df_main_valid['RPOGX']=(datafile['FPOGX']-0.5)*1920       
    df_main_valid['RPOGY']=-1*(datafile['FPOGY']-0.5)*1080
    df_main_valid['LPOGX']=(datafile['FPOGX']-0.5)*1920       
    df_main_valid['LPOGY']=-1*(datafile['FPOGY']-0.5)*1080
    
    return datafile

convert_gaze_coords(df_main_valid) #example


 
outiesY=df_main_valid[df_main_valid['FPOGY']>530]
outies_Y=df_main_valid[df_main_valid['FPOGY']<-530]
outiesX=df_main_valid[df_main_valid['FPOGY']>950]
outies_X=df_main_valid[df_main_valid['FPOGY']<-950]



#Keep gaze samples with valid flag
fixation_valid_dataset=dataset.query("FPOGV==1")
best_valid_dataset=dataset.query("BPOGV==1")


#List with images numbering
high_list=[1,2,7,12,13,15,19,25,27,29,41,42,43,44,48,49,51,54,55,59,61,64,67, \
           74,76,77,84,87,88,91,94,95,100,101,112,113]
low_list=[3,6,10,17,21,23,28,33,35,38,39,40,46,50,52,58,60,62,63,70,72,73,75, \
          78,80,82,89,90,92,97,99,102,103,104,105,108]

    
for j, i in itertools.product(low_list, range(0,4)):
    filtered=df_main.query("USER.str.startswith('S"+str(j)+".JPG_LOW_"+str(i)+"_').to_numpy()")
    #m = filtered.loc[: filtered['CS'].eq(1).idxmax()]
    m = filtered['CS'].eq(1)
    df1 = filtered.loc[: m.idxmax()] if m.any() else pd.DataFrame()
    #rt=m.iloc[-1,1]-m.iloc[0,1]
    print('Image_'+str(j),'Block_'+str(i), df1)


df_main_valid['CS'].value_counts()

#Draw fixation map 
for j, i in itertools.product(low_list, range(0,4)):
    filtered=df_main_valid.query("USER.str.startswith('S"+str(j)+".JPG_LOW_"+str(i)+"_').to_numpy()")  
    x=filtered.groupby('FPOGID')['FPOGX'].mean()
    y=filtered.groupby('FPOGID')['FPOGY'].mean()
    #sum_x=filtered.groupby('FPOGID')['FPOGX'].sum()
    #sum_y=filtered.groupby('FPOGID')['FPOGY'].sum()
    #len_x=filtered.groupby('FPOGID')['FPOGY'].count()
    #len_y=filtered.groupby('FPOGID')['FPOGY'].count()
    #centroid_x=sum_x/len_x
    #centroid_y=sum_y/len_y
    fix_dur=filtered.groupby('FPOGID')['FPOGD'].max()
    fig, ax = plt.subplots(figsize=(20, 11))
    ax.scatter(x,y,zorder=1,marker='o',s=fix_dur*10000, color='lime',alpha=0.5)
    abs_path=scene_folder
    img = plt.imread(abs_path+"S"+str(j)+".jpg")
    plt.imshow(img, extent=[-960, 960, -540, 540],aspect='auto') 
    plt.show()
    print(j,i)
    plt.close()    


#Find first fixation duration for each image
for j, i in itertools.product(low_list, range(0,4)):
    with open ('participant2.csv','w') as csvfile:
    filtered=df_main_valid.query("USER.str.startswith('S"+str(j)+".JPG_LOW_"+str(i)+"_').values") 
    ffd=filtered.groupby('FPOGID')['FPOGD'].max()
    first_fix_dur=ffd.iloc[[1]]
    print (first_fix_dur,'scene_'+str(j),'block_'+str(i))


#Draw scanpath 
for j, i in itertools.product(high_list, range(0,4)):
    filtered=df_main_valid.query("USER.str.startswith('S"+str(j)+".JPG_HIGH_"+str(i)+"_').values")    
    x=filtered.groupby('FPOGID')['FPOGX'].mean()
    y=filtered.groupby('FPOGID')['FPOGY'].mean()
    fix_dur=filtered.groupby('FPOGID')['FPOGD'].max()
    fig, ax = plt.subplots(figsize=(20, 11))
    ax.scatter(x,y,zorder=1,marker='o',s=fix_dur*10000, color='lime',alpha=0.5)
    ax.plot(x,y,'-o',linewidth=3,color='blue')
    abs_path=scene_folder
    img = plt.imread(abs_path+"S"+str(j)+".jpg")
    plt.imshow(img, zorder=0, extent=[-960, 960, -540, 540],aspect='auto')
    for i in range(len(fix_dur)):
        ax.annotate(str(i+1), 
        xy=(fix_dur.iloc[i],fix_dur.iloc[i]), 
        xytext=(x.iloc[i], y.iloc[i]), 
        fontsize=30, 
        color='black',
        ha='center',
        va='center')
    plt.show()
    plt.close()

#Draw heatmap
filtered=best_valid_dataset.query("USER.str.startswith('S55.JPG_HIGH_').values")
fig, ax = plt.subplots(figsize=(25, 14))
abs_path=scene_folder
img = plt.imread(abs_path+'S55.jpg')
sns.kdeplot(data=filtered, 
            x='FPOGX',
            y='FPOGY',
            bw_method=0.4,
            bw_adjust=0.9,
            cbar=True,
            levels=101,
            shade=True,
            cmap=cm.jet, 
            alpha=0.9)
plt.imshow(img, zorder=0, extent=[-960, 960, -540, 540],aspect='auto')
plt.show()
plt.close()







import numpy as np
# Width and height of the image.
W = 5
H = 5
# Generate a random image.
x = np.random.rand(W, H) > 0.5
# Print the image.
print (x)
# Print the iamge as bit cells.
print ((x == 1).astype(int))
# Print the centroid.
xis, yis = np.nonzero(x)
print (xis.mean(), yis.mean())
np.mean(x.astype(int), axis=0)
























##Ninis dataset
import pandas as pd
dataset=pd.read_csv('customerbuysbackup.txt', sep='\t', header=None, names=['id','date','prod1','prod2',
                                                                            'prod3','prod4','prod5','prod6',
                                                                            'prod7','prod8','prod9','prod10',
                                                                            'prod11'])

idate=dataset.iloc[:,0:2]
dataset.drop(['date','id'],inplace=True, axis=1)
dataset= dataset.unstack().reset_index()

dataset = dataset.pivot_table(index = 0, columns = 'level_1', aggfunc = "count", fill_value = 0).T.reset_index()
dataset.drop(['level_0', 'level_1'], axis = 1, inplace = True)

dataset.insert(0, 'ID', idate.id, True)
dataset.insert(1, 'DATE', idate.date, True)

dataset.to_csv('Market_basket_analysis.csv', index=False)



























filtered=dataset.query(" USER.str.startswith('S6').values ")
x=filtered.BPOGX
y=filtered.BPOGY
#plt.scatter(x,y,zorder=2,marker='o',s=5, color='blue',alpha=0.7)
plt.plot(x,y,'-o')
img = plt.imread(r'C:\Users\presi\Desktop\VS_Task\Scenes\S6.jpg')
plt.imshow(img, zorder=0, extent=[-960, 960, -540, 540])
plt.show()
plt.close()




valid_samples=et.query('FPOGX!=0')
invalid_samples=et.query('FPOGX==0')

filter = np.where(dataset['USER']=='P6_START')

dataset.loc[filter]


filtered=dataset.query(" USER.str.startswith('P6').values ")
filtered=filtered.iloc[2:]
start_trial=filtered.TIME.iloc[0]
end_trial=filtered.TIME.iloc[-1]
duration_trial=end_trial-start_trial

filtered7=dataset.query(" USER.str.startswith('P7').values ")
start_trial7=filtered7.TIME.iloc[0]
end_trial7=filtered7.TIME.iloc[-1]
duration_trial7=end_trial7-start_trial7

filtered8=dataset.query(" USER.str.startswith('P8').values ")
start_trial8=filtered8.TIME.iloc[0]
end_trial8=filtered8.TIME.iloc[-1]
duration_trial8=end_trial8-start_trial8

filtered2=dataset.query(" USER.str.startswith('P2').values ")
start_trial2=filtered2.TIME.iloc[0]
end_trial2=filtered2.TIME.iloc[-1]
duration_trial2=end_trial2-start_trial2

filtered3=dataset.query(" USER.str.startswith('P3').values ")
start_trial3=filtered3.TIME.iloc[0]
end_trial3=filtered3.TIME.iloc[-1]
duration_trial3=end_trial3-start_trial3