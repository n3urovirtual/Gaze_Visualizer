import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import time
import itertools
import csv

#Read the raw ET file
dataset=pd.read_csv('M_1_2021_Mar_13_1417.tsv', sep='\t')
scene_folder=r'C:\Users\presi\Desktop\Memory guided attention in cluttered '\
    'scenes v.3\Experimental Tasks\Task1\Scenes'

df=dataset[['BPOGX', 'BPOGY']]
df.to_csv('L1_datafile.tsv',sep='\t', index=False,header=False)


#Calculate sampling rate for the entire dataset. Applied before any analysis
def sampling_rate(datafile):
    datafile['Sampling_Rate']=datafile['TIME'].diff()

    return datafile['Sampling_Rate'].describe()

sampling_rate(dataset) #example


#Calculate cumulative time starting from 0
def cumulative_time(datafile):
    datafile['Cumulative_time']=datafile['Sampling_Rate'].cumsum().fillna(0)
    
    return datafile

cumulative_time(dataset)

# Keep only samples during scene viewing (main trials)
def keep_trials(datafile):
    global df
    #df=datafile[(datafile['IMAGE'].astype(str).str.startswith('S')) & 
                #(datafile['START_END']=='START')]
    df=datafile[datafile['USER'].astype(str).str.startswith('S')]
                
    return df

keep_trials(dataset) #example


new=df['USER'].str.split("_", -1, expand=True)
df['IMAGE']=new[0]
df['CLUTTER']=new[1]
new_2=df['IMAGE'].str.split(".",1,expand=True)
new_2[0]=new_2[0].str.replace("S","", 1)
new_2[0].astype(int)
df['IMAGE']=new_2[0]


#Keep only trials for which a valid recording exists
def keep_valid_recs(datafile):
    global df_valid
    df_valid=datafile[datafile['BPOGV']==1]

    return df_valid

keep_valid_recs(df) #example


#Convert gaze data from gp3 coordinates to psychopy coordinates
def convert_gaze_coords(datafile):
    datafile['BPOGX']=(datafile.loc[:,'BPOGX']-0.5)*1920
    datafile['BPOGY']=-1*(datafile.loc[:,'BPOGY']-0.5)*1080
    datafile['FPOGX']=(datafile.loc[:,'FPOGX']-0.5)*1920
    datafile['FPOGY']=-1*(datafile.loc[:,'FPOGY']-0.5)*1080
    datafile['RPOGX']=(datafile.loc[:,'RPOGX']-0.5)*1920
    datafile['RPOGY']=-1*(datafile.loc[:,'RPOGY']-0.5)*1080
    datafile['LPOGX']=(datafile.loc[:,'LPOGX']-0.5)*1920
    datafile['LPOGY']=-1*(datafile.loc[:,'LPOGY']-0.5)*1080

    return datafile

convert_gaze_coords(df) #example


#Convert gaze data from gp3 coordinates to gazepath coordinates
def convert_gaze_coords_alt(datafile):
    datafile['BPOGX']=datafile.loc[:,'BPOGX']*1920
    datafile['BPOGY']=datafile.loc[:,'BPOGY']*1080
    datafile['FPOGX']=datafile.loc[:,'FPOGX']*1920
    datafile['FPOGY']=datafile.loc[:,'FPOGY']*1080
    datafile['RPOGX']=datafile.loc[:,'RPOGX']*1920
    datafile['RPOGY']=datafile.loc[:,'RPOGY']*1080
    datafile['LPOGX']=datafile.loc[:,'LPOGX']*1920
    datafile['LPOGY']=datafile.loc[:,'LPOGY']*1080

    return datafile

convert_gaze_coords_alt(df_valid) #example



df_valid['LEYEZ']=df_valid['LEYEZ']*1000
df_valid['REYEZ']=df_valid['REYEZ']*1000

df_valid.to_csv('sub_1_memory.tsv',sep='\t',index=False)
# def cyclopic_vision(datafile):
#     df_main_valid['CPOGX']=(datafile['LPOGX']+datafile['RPOGX'])/2
#     df_main_valid['CPOGY']=(datafile['LPOGY']+datafile['RPOGY'])/2
    
#     return datafile

# cyclopic_vision(df_valid) #example



#Keep samples within screen range (x-10, y-10)
def keep_in_screen_samples(datafile, X_upper, X_lower, Y_upper, Y_lower):
    global d1
    
    d1=datafile[(datafile['BPOGX']<X_upper) & (datafile['BPOGX']>X_lower) & \
                (datafile['BPOGY']<Y_upper) & (datafile['BPOGY']>Y_lower)]
    
    return d1, d1.BPOGX.min(), d1.BPOGX.max(), d1.BPOGY.min(), d1.BPOGY.max()

keep_in_screen_samples(df, 950, -950, 530, -530) #example

#Keep gaze samples with valid flag
fixation_valid_dataset=dataset.query("FPOGV==1")
best_valid_dataset=dataset.query("BPOGV==1")



#Apply savitzky-golay filter
from scipy.signal import savgol_filter
bpogx=df_main_valid['BPOGX']
bpogy=df_main_valid['BPOGY']
df_main_valid['SAV_BPOGX']=savgol_filter(bpogx, 11, 2)
df_main_valid['SAV_BPOGY']=savgol_filter(bpogy, 11, 2)


x=df_main_valid['SAV_BPOGX'].to_numpy()
y=df_main_valid['SAV_BPOGY'].to_numpy()
sr=df_main_valid['Sampling_Rate'].to_numpy()

#Pixels to radians-degrees
from math import atan, atan2, tan, pi,degrees
Size=480
Distance=600
screen_size_in_rad=2*atan(Size/(2*Distance))
screen_in_deg=screen_size_in_rad* (180/pi)




screen_size=48.5
viewing_distance=60
screen_resolution=1920
px2deg=degrees(atan2(.5 * screen_size, viewing_distance)) / (.5 * screen_resolution)

def deg_per_pixel(screen_size, viewing_distance, screen_resolution):
    """Determine `px2deg` factor for EyegazeClassifier
    Parameters
    ----------
    screen_size : float
      Either vertical or horizontal dimension of the screen in any unit.
    viewing_distance : float
      Viewing distance from the screen in the same unit as `screen_size`.
    screen_resolution : int
      Number of pixels along the dimensions reported for `screen_size`.
    """
    return degrees(atan2(.5 * screen_size, viewing_distance)) / \
        (.5 * screen_resolution)


#def _get_velocities(self, data):
    # euclidean distance between successive coordinate samples
    # no entry for first datapoint!
    
velocities = (np.diff(x) ** 2 + np.diff(y) ** 2) ** 0.5
    # convert from px/sample to deg/s
velocities *= px2deg * sr
    #turn velocities
df_main_valid['Distance']=((df_main_valid['SAV_BPOGX']-df_main_valid['SAV_BPOGX'].shift())**2 + (df_main_valid['SAV_BPOGY']-df_main_valid['SAV_BPOGY'].shift())**2)**0.5
df_main_valid['IS_TIME']=df_main_valid['TIME']-df_main_valid['TIME'].shift()
df_main_valid['IS_TIME']=df_main_valid['IS_TIME']/1000


df_main_valid['Velocity']= df_main_valid['Distance']/df_main_valid['IS_TIME']


intdist= (np.diff(x)**2 + np.diff(y)**2)**0.5
# get inter-sample times
inttime = np.diff(time)
# recalculate inter-sample times to seconds
#inttime = inttime / 1000.0

vel = intdist / inttime
acc = np.diff(vel)


distance=df_main_valid['Distance'].to_numpy()

from math import atan,pi
radians=2*np.arctan(distance/(2*2400))
degrees=radians*(180/pi)


df_main_valid['velocity']=degrees/df_main_valid['IS_TIME']

df_main_valid['acceleration']=df_main_valid['velocity']-df_main_valid['velocity'].shift()/df_main_valid['IS_TIME']

unphysiological=df_main_valid.velocity[df_main_valid.velocity>1000]

saccade_filter=df_main_valid.velocity[df_main_valid.velocity>35]

df_main_valid.plot(kind='line',x=df_main_valid['TIME_2'],y=df_main_valid['velocity'])


df_main_valid['TIME_2']=df_main_valid['IS_TIME'].cumsum()


#List with images numbering
high_list=[1,2,7,12,13,15,19,25,27,29,41,42,43,44,48,49,51,54,55,59,61,64,67, \
           74,76,77,84,87,88,91,94,95,100,101,112,113]
low_list=[3,6,10,17,21,23,28,33,35,38,39,40,46,50,52,58,60,62,63,70,72,73,75, \
          78,80,82,89,90,92,97,99,102,103,104,105,108]

img_id=[1,2,7,12,13,15,19,25,27,29,41,42,43,44,48,49,51,54,55,59,61,64,67, 
      74,76,77,84,87,88,91,94,95,100,101,112,113,3,6,10,17,21,23,28,33,
      35,38,39,40,46,50,52,58,60,62,63,70,72,73,75,78,80,82,89,90,92,97,
      99,102,103,104,105,108]

for j, i in itertools.product(low_list, range(0,4)):
    filtered=df_main.query("USER.str.startswith('S"+str(j)+".JPG_LOW_"+str(i)+"_').values()")
    #m = filtered.loc[: filtered['CS'].eq(1).idxmax()]
    m = filtered['CS'].eq(1)
    df1 = filtered.loc[: m.idxmax()] if m.any() else pd.DataFrame()
    #rt=m.iloc[-1,1]-m.iloc[0,1]
    print('Image_'+str(j),'Block_'+str(i), df1)


df_main_valid['CS'].value_counts()


#Draw BPOG scatterplot
for j, i in itertools.product(img_id, range(0,4)):
    filtered=df_main_valid.query("IMAGE.str.startswith('S"+str(j)+"').values & BLOCK=="+str(i)+"")
    x=filtered['SAV_BPOGX']
    y=filtered['SAV_BPOGY']
    fig, ax = plt.subplots(figsize=(20, 11))
    ax.scatter(x,y,zorder=1,marker='o', color='lime',alpha=0.5)
    abs_path=scene_folder
    img = plt.imread(abs_path+"\S"+str(j)+".jpg")
    plt.imshow(img, extent=[-960, 960, -540, 540],aspect='auto')
    plt.show()
    print(j,i)
    plt.close()
    

#Draw fixation map
for j, i in itertools.product(low_list, range(0,4)):
    filtered=df_main_valid.query("USER.str.startswith('S"+str(j)+".JPG_LOW_"+str(i)+"_').values")
    x=filtered.groupby('FPOGID')['FPOGX'].mean()
    y=filtered.groupby('FPOGID')['FPOGY'].mean()
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
for j, i in itertools.product(low_list, range(0,4)):
    filtered=df_main_valid.query("USER.str.startswith('S"+str(j)+".JPG_LOW_"+str(i)+"_').values")
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


