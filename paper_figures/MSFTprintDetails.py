import pandas as pd
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as pl
import os
import itertools
import glob
import datetime

path='/project/comcast-ping/stabledist/geolocResultados'
os.chdir(path)

ASnames_df=pd.read_csv('/project/comcast-ping/stabledist/COMCASTTransitProviders/inputData/ASN_ASname.csv',header='infer')
cc2rir_df=pd.read_csv('/project/comcast-ping/stabledist/geoloc/cc2rir.csv',sep=',')

CPs_v=[8075, ]

TIER1_set=set()

for CP in CPs_v:
    for YYYY in range(1998,2003):
        for mm in range(1,13):
            if len(glob.glob('BGP_graphs/%s%.2d00.net'%(YYYY,mm)))!=0:
                peers_df=pd.read_csv('BGP_graphs/%s%.2d00.net'%(YYYY,mm),sep=' ',names=['AS1','AS2'])
                CPPeers_df=peers_df.loc[(peers_df['AS1']==CP) | (peers_df['AS2']==CP)][['AS1','AS2']].values
                peers_set=set(CPPeers_df.reshape(CPPeers_df.size,).tolist())
                for ASN in peers_set:
                    try:
                        print "%s\t%s\t%s\t%s"%(YYYY,mm,ASN,ASnames_df.loc[ASnames_df['ASN']==ASN]['ASname'].values[0])
                        TIER1_set.add((ASN,ASnames_df.loc[ASnames_df['ASN']==ASN]['ASname'].values[0]))
                    except:
                        currentCounters=0


for S in TIER1_set:
    print S         