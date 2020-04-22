# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 09:57:12 2020

@author: User
"""


import pandas as pd
import numpy as np


def daten_laden(dataset):
    
    df = pd.read_csv(f"Data/daten{dataset}P.csv")
    return df

def extremwerte_spalten(zeilenindex,df):
    maxi = np.zeros((1,15))
    mini = np.zeros((1,15))
    for j in range(7,22):
        maxi[0,j-7] = max(df.iloc[zeilenindex,j])
        mini[0,j-7] = min(df.iloc[zeilenindex,j])
    return [maxi, mini]


dataset = 1
df = daten_laden(dataset)

groesse = int((df.shape[0]-1)/100)
extremwerte = np.zeros((2*groesse+1,15))
extremwerte[0,:] = df.iloc[0,7:22]
for i in range(0,groesse):
    zeilenindex = range(100*i+1,100*i+101)
    [maxi,mini] = extremwerte_spalten(zeilenindex, df)
    extremwerte[2*i+1,:] = mini
    extremwerte[2*i+2,:] = maxi

extremwerte_prozent = np.round(extremwerte[:,:]/extremwerte[0,:]-1,4)

