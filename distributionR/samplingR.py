
import pandas as pd
import numpy as np
import os
import random

path = os.path.split(os.path.realpath(__file__)) [0]

dataR = pd.read_csv(os.path.join(path,'AA_rotamer_list'),header=None,sep='\s+')
interval = np.linspace(-180,180,49)
def samplingr(res,chi,chi_class):
    #interval = np.linspace(-180,180,49)
    rotamer_list = dataR[ (dataR.iloc[:,0] == res) & (dataR.iloc[:,1] == chi) & (dataR.iloc[:,2] == chi_class) ].iloc[0,:].to_list()
    rn = random.choices([i for i in range(15)],weights = rotamer_list[3:], k=1)[0]
    mini_interval = np.linspace( interval[chi_class], interval[ chi_class+1], 16)
    angle = np.random.uniform(mini_interval[rn], mini_interval[rn+1],1)[0]
    return angle

