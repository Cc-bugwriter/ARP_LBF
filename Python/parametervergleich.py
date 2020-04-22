# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:26:08 2020

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt

def daten_laden(dataset):
    
    df = pd.read_csv(f"Data/daten{dataset}P.csv")
    return df

dataset = 1
df = daten_laden(dataset)

"Plot der Abhängigkeit der drei Eigenfrequenzen von den Massen"
"""
plt.figure()
plt.subplot(131)
plt.plot(df.loc[1:100,'m2'],df.loc[1:100,'omega_1'],'ro')
plt.plot(df.loc[101:200,'m3'],df.loc[101:200,'omega_1'],'bo')
plt.plot(df.loc[201:300,'m4'],df.loc[201:300,'omega_1'],'go')
plt.subplot(132)
plt.plot(df.loc[1:100,'m2'],df.loc[1:100,'omega_2'],'ro')
plt.plot(df.loc[101:200,'m3'],df.loc[101:200,'omega_2'],'bo')
plt.plot(df.loc[201:300,'m4'],df.loc[201:300,'omega_2'],'go')
plt.subplot(133)
plt.plot(df.loc[1:100,'m2'],df.loc[1:100,'omega_3'],'ro')
plt.plot(df.loc[101:200,'m3'],df.loc[101:200,'omega_3'],'bo')
plt.plot(df.loc[201:300,'m4'],df.loc[201:300,'omega_3'],'go')
"""

"Plot der Abhängigkeit der Eigenfrequenzen von den Steifigkeiten"

plt.figure()
plt.subplot(131)
plt.plot(df.loc[301:400,'k5'],df.loc[301:400,'omega_1'],'ro')
plt.plot(df.loc[401:500,'k6'],df.loc[401:500,'omega_1'],'bo')
plt.subplot(132)
plt.plot(df.loc[301:400,'k5'],df.loc[301:400,'omega_2'],'ro')
plt.plot(df.loc[401:500,'k6'],df.loc[401:500,'omega_2'],'bo')
plt.subplot(133)
plt.plot(df.loc[301:400,'k5'],df.loc[301:400,'omega_3'],'ro')
plt.plot(df.loc[401:500,'k6'],df.loc[401:500,'omega_3'],'bo')

"Plot der Abhängigkeit der Dämpfung von den Steifigkeiten"
plt.figure()
plt.subplot(131)
plt.plot(df.loc[301:400,'k5'],df.loc[301:400,'D_1'],'ro')
plt.plot(df.loc[401:500,'k6'],df.loc[401:500,'D_1'],'bo')
plt.subplot(132)
plt.plot(df.loc[301:400,'k5'],df.loc[301:400,'D_2'],'ro')
plt.plot(df.loc[401:500,'k6'],df.loc[401:500,'D_2'],'bo')
plt.subplot(133)
plt.plot(df.loc[301:400,'k5'],df.loc[301:400,'D_3'],'ro')
plt.plot(df.loc[401:500,'k6'],df.loc[401:500,'D_3'],'bo')

"Plot der Abhängigkeit der Eigenvektoren von den Steifigkeiten"
plt.figure()
plt.subplot(331)
plt.plot(df.loc[301:400,'k5'],df.loc[301:400,'EVnorm1_1'],'ro')
plt.plot(df.loc[401:500,'k6'],df.loc[401:500,'EVnorm1_1'],'bo')
plt.subplot(332)
plt.plot(df.loc[301:400,'k5'],df.loc[301:400,'EVnorm1_2'],'ro')
plt.plot(df.loc[401:500,'k6'],df.loc[401:500,'EVnorm1_2'],'bo')
plt.subplot(333)
plt.plot(df.loc[301:400,'k5'],df.loc[301:400,'EVnorm1_3'],'ro')
plt.plot(df.loc[401:500,'k6'],df.loc[401:500,'EVnorm1_3'],'bo')
plt.subplot(334)
plt.plot(df.loc[301:400,'k5'],df.loc[301:400,'EVnorm2_1'],'ro')
plt.plot(df.loc[401:500,'k6'],df.loc[401:500,'EVnorm2_1'],'bo')
plt.subplot(335)
plt.plot(df.loc[301:400,'k5'],df.loc[301:400,'EVnorm2_2'],'ro')
plt.plot(df.loc[401:500,'k6'],df.loc[401:500,'EVnorm2_2'],'bo')
plt.subplot(336)
plt.plot(df.loc[301:400,'k5'],df.loc[301:400,'EVnorm2_3'],'ro')
plt.plot(df.loc[401:500,'k6'],df.loc[401:500,'EVnorm2_3'],'bo')
plt.subplot(337)
plt.plot(df.loc[301:400,'k5'],df.loc[301:400,'EVnorm3_1'],'ro')
plt.plot(df.loc[401:500,'k6'],df.loc[401:500,'EVnorm3_1'],'bo')
plt.subplot(338)
plt.plot(df.loc[301:400,'k5'],df.loc[301:400,'EVnorm3_2'],'ro')
plt.plot(df.loc[401:500,'k6'],df.loc[401:500,'EVnorm3_2'],'bo')
plt.subplot(339)
plt.plot(df.loc[301:400,'k5'],df.loc[301:400,'EVnorm3_3'],'ro')
plt.plot(df.loc[401:500,'k6'],df.loc[401:500,'EVnorm3_3'],'bo')

