import pandas as pd
import numpy as np

def get_all_trajs(df,pattern='S([0-9]+)'):
    """get_all_trajs(df)
    Get station numbers to mark which columns are for which station
    Input: df - pandas dataframe
    """
    col=df.columns
    #get maximum station number
    #snum=col.str.extract(pattern).astype(float)
    stat_id=get_station_ind(df,pattern)
    nrow=len(df)
    ind=[]
    #ind=[get_row_traj(row,stat_id) for row in ]
    #Slow as a frozen snail, but it works
    for i in range(nrow):
        row = df.iloc[i]
        i0=get_row_traj(row,stat_id)
        ind.append(i0.tolist())
    #df.assign('traj'=ind)
    return ind


def get_station_ind(df,pattern='S([0-9]+)'):
    """get_station_ind(df)
    Get station numbers to mark which columns are for which station
    Input: col - Pandas index for column names
    """
    col=df.columns
    stat_dict=dict()
    #get maximum station number
    stats=col.str.extract(pattern,expand=False).astype(float)
    #drop last index
    stats=stats[:-1].astype(int)

    num_max=stats.max()
    num_old=np.nan
    stat_id=[]
    #Find stations by finding column indices when station number changes.
    #i.e. use first feature per station as test to find trajectories.
    for i,num in enumerate(stats):
        if num != num_old:
           #print(num,num_old)
           stat_id.append([num,i])
           # stat_id[num,0]=num
           # stat_id[num,1]=i
           num_old=num
    return np.array(stat_id).astype(int)


def get_row_traj(row,stat_id):
    #want a list of stations which are nonzero on a row.
    msk= ~row.isnull().values
    #indices that are present
    sub_msk=msk[stat_id[:,1]]
    i0=stat_id[sub_msk,0]
    return i0


def get_unique_trajs(ind):
    """get_unique_trajs
    Build up list of unique trajectories through the system, and their count.
    Input: ind - list of each rows trajectory through the system.
    Output traj_list - list of unique trajectories
           traj_count - corresponding counts.
    """
    traj_list=[]

    Ntraj=0
    traj_count=[]
    for ar in ind:
        try:
            i=traj_list.index(ar)
            traj_count[i]+=1
        except:
            print('adding_new traj')
            traj_list.append(ar)
            traj_count.append(1)
    return traj_list, traj_count

def get_all_unique_traj(df):
    """actually computes all trajectories through system,
    and their counts"""
    #For all features - agrees with Borthwick
    ind=get_station_traj(df,pattern='F([0-9]+)')
    traj_list,traj_count=get_unique_traj(ind)
    print(len(traj_list))
    #For all stations - does not agree with Borthwick, Gott
    ind=get_station_traj(df,pattern='S([0-9]+)')
    traj_list,traj_count=get_unique_traj(ind)
    print(len(traj_list))
    
    
