# Script to load data and get handle on it.
# Perhaps even trajectories for parts.

# 1. Sparsity?
# Most important feature - find highest correlation.
# Unregularized linear regression, find largest coefficient.  
# 2. RDMS structure?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 
import re

data_dir='small_data'

def load_data(data_dir='small_data',nrows=1000):
    df_cat=pd.read_csv(data_dir+'/train_cat.csv',nrows=nrows,dtype=object,index_col=0)
    df_date=pd.read_csv(data_dir+'/train_date.csv',nrows=nrows,index_col=0)
    df_numeric=pd.read_csv(data_dir+'/train_numeric.csv',nrows=nrows,index_col=0)        
    return df_cat, df_date,df_numeric

def load_numeric_data(data_dir='small_data',nrows=1000):
    df_numeric=pd.read_csv(data_dir+'/train_numeric.csv',nrows=nrows,index_col=0)        
    return df_numeric

def plot_dates(df_date):
    print(df_date.shape)
    date_min=df_date.iloc[:,:-2].min()
    date_max=df_date.iloc[:,:-2].max()    
    date_mean=df_date.iloc[:,:-2].median()
    
    msk = ~np.isnan(date_min)
    plt.figure()
    plt.title('Mean')
    plt.hist(date_mean[msk])
    plt.show(block=False)

    plt.figure()
    plt.hist(date_min[msk])
    plt.title('Min')
    plt.show(block=False)

    plt.figure()
    plt.title('Max-Min')
    plt.hist(date_max[msk]-date_min[msk])
    plt.show(block=False)
    print(sum(msk))
    
#1. get station numbers.
#2. get column, label pairs.
#3. sort on date (based on almost sorted)
#def make_traj_name():

def get_line(col):
    """get_line(df)
    Get line numbers to mark which columns are for which line.
    Hardcoded to use from 1-3 Lines
    Input: col - Pandas index for column names
    """
    nr = np.arange(len(col))
    for i in range(4):
        msk=df.columns.str.contains('L{}'.format(i))

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
    for i,num in enumerate(stats):
        if num != num_old:
           #print(num,num_old)
           stat_id.append([num,i])
           # stat_id[num,0]=num
           # stat_id[num,1]=i
           num_old=num
    return np.array(stat_id).astype(int)

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
        #print(i0)
    #df.assign('traj'=ind)
    return ind

def get_unique_trajs(ind):
    """get_unique_trajs
    Build up list of unique trajectories through the system, and their count.
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

def get_row_traj(row,stat_id):
    #want a list of stations which are nonzero on a row.
    msk= ~row.isnull().values
    #indices that are present
    sub_msk=msk[stat_id[:,1]]
    i0=stat_id[sub_msk,0]
    return i0
    
def pca(df):
    """pca(df)
    Create PCA_model
    Input df - pandas dataframe (where last column is response)
    Return PCA -
           mat - transformed matrix
    """
    mat=df.fillna(df.mean()).values;
    mat=mat[:,:-2]
    #fill any NaNs with 0
    msk=np.isnan(mat)
    mat[msk]=0
    PCA_model=PCA(n_components=10,niter=20)
    PCA_model.fit_transform(mat)
    return PCA_model,mat

def cluster(mat):
    KMeans_model=KMeans(clusters=3)
    KMeans_model.fit(mat)
    return KMeans_model

#Look into random forest model

from sklearn.ensemble import RandomForestClassifier

def fit_RFC(df):
    nrow,ncol=df.shape
    RFC_model = RandomForestClassifier()
    RFC_model.fit()

    

if __name__=="__main__":
    df_num=load_numeric_data(data_dir,nrows=None)
    traj=get_all_trajs(df_num)
    
