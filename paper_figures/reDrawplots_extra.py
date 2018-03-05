import pandas as pd
import numpy as np
import scipy.stats
import matplotlib
import matplotlib.pyplot as pl
import os

path='/Users/estcarisimo/Doctorado/2017/MovimientoCDNs/EvolCoresNorm'
os.chdir(path)

dataSources=('Ark_BGP',)
for dataSource in dataSources:
    
    normalizedKcore_df=pd.read_csv('%s/cores_extra.csv'%dataSource, header='infer')
    
    fig, ax1 = pl.subplots(1,figsize=(10, 5))
    ax1.yaxis.grid(True, linestyle='-', color='#bababa',alpha=0.5)
    ax1.xaxis.grid(True, linestyle='-', color='#bababa',alpha=0.5)
    for CP in np.sort(list(normalizedKcore_df.columns.values)[:-2]):
        ax1.plot(\
                    normalizedKcore_df['time'].values,\
                    normalizedKcore_df[CP].values/normalizedKcore_df['top_core'].values.astype(float),\
                    label=CP,\
                    linewidth=2,\
                    alpha=0.7\
                    )
    
    ax1.set_yticks(np.arange(0,1.01,0.1),minor=False)
    ticks=[]
    xlabels=[]
    for Y in np.arange(199900,201801,100):
        xlabels.append(int(Y/100))
        ticks.append(Y)
    
    ax1.set_xticks(ticks,minor=False)
    ax1.set_xticklabels(xlabels,rotation=90, minor=False,ha='center')
    
    ax1.axis([199800,201800,0,1])
    if dataSource=='Ark_BGP':
        ax1.set_title('core evolution of CPs in Ark+BGP dataset',fontsize=24)
    else:
        ax1.set_title('core evolution of CPs in %s dataset'%dataSource,fontsize=24)  
    ax1.set_ylabel('normalized core',fontsize=20)
    ax1.set_xlabel('',fontsize=20)
    ax1.tick_params(labelsize=20)
    ax1.legend(loc='upper left',ncol=1,frameon=False,fontsize=17)
    fig.subplots_adjust(hspace=0)
    fig.tight_layout()
    #pl.show()
    pl.savefig('%s/CPsExtraKcoreEvolution.pdf'%dataSource)