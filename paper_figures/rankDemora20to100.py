import pandas as pd
import numpy as np
import scipy.stats
import matplotlib
import matplotlib.pyplot as pl
import os
import matplotlib.cm as cm
from matplotlib.colors import LogNorm


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

##############################################################################
'''
    Seccion de carga de archivos provenientes de procesamiento previo, ASnames y AStype.
'''


# Seleccion de directorio

path='/Users/estcarisimo/Doctorado/2017/MovimientoCDNs/'
os.chdir(path)

# Carga de archivos

Kcore_df=pd.read_csv('EvolCoresNorm/BGP/kcores_completo.csv', header='infer') # Output LaNet-vi
ASnames_df=pd.read_csv('EvolCoresNorm/ASN_ASname.csv',header='infer')       #ASN--> ASnames (CAIDA?)
ASclass_df=pd.read_csv('makeRanking/20150801.as2types.txt',header='infer',sep='|') #ASN--> type (CAIDA: http://data.caida.org/datasets/as-classification/)

#Normalizacion de los k-cores
M=np.matrix(Kcore_df.values)
Mn=M[:,2:-1]/M[:,1]

##############################################################################
'''
    Seccion aplicacion del filtro. Se agregan otras alternativas previamente formuladas
    Criterio usado:
        + Que el AS haya estado en el k-core en algun momento de la historia
        + Que haya estado en el kcore>=0.95 en los ultimos 6 meses
        
'''

#MAXX=(Mn.max(0)==1) # Que hayan estado en el TOPcore
#MAXX=(Mn[-1,:]==01) # Que hayan finalizado en el TOPcore
# Que hayan estado en el TOPcore y que haya estado en el kcore>=0.95 en los ultimos 6 meses
MAXX=(Mn[-6:,:].max(0)>=0.95) & (Mn.max(0)==1) 
ASesTOP=np.where(MAXX.tolist()[0])

VelocidadLlegadaTOPcore=[]

for AS in ASesTOP[0].tolist():
    VelocidadLlegadaTOPcore.append((AS+1,M[:,-1][(Mn[:,AS]==1)].min()-M[:,-1][(Mn[:,AS]>0.3)].min(),M[:,-1][(Mn[:,AS]>0.3)].min(),M[:,-1][(Mn[:,AS]==1)].min()))
   
 
V_df=pd.DataFrame(VelocidadLlegadaTOPcore,columns=['ASN','demora','inicio','llegada'])

##############################################################################
'''
    Codigo irrelevante pero sintaxis util para algunos analisis breves
'''

#V_df.loc[V_df['demora']>0].sort('demora',ascending=False)

##Ranking de velocidad
#
##Netflix (61)
#V_df.loc[V_df['demora']>0].sort('demora',ascending=True).loc[V_df['ASN']==2906].index
#
##Apple (15)
#V_df.loc[V_df['demora']>0].sort('demora',ascending=True).loc[V_df['ASN']==714].index
#
##Google (78)
#V_df.loc[V_df['demora']>0].sort('demora',ascending=True).loc[V_df['ASN']==15169].index
#
##Facebook (112)
#V_df.loc[V_df['demora']>0].sort('demora',ascending=True).loc[V_df['ASN']==32934].index
#
##MSFT (1)
#V_df.loc[V_df['demora']>0].sort('demora',ascending=True).loc[V_df['ASN']==8075].index

##############################################################################
'''
    Valores medios para entender segunda parte del procesamiento.
    Conclusion: 
        Gran cantidad de ASes llegan al TOPcore luego de 2011.
        Esto se debe a que post 2011 existe una gran cantidad de facilidades tecnologicas
        para hacerse visible rapidamente a bajo costo.
        En simultaneo, a partir de 2011 la cantidad de puntos de observacion es alta
'''


print "Anio promedio de inicio: %s"%np.mean(V_df.loc[V_df['demora']>0][['inicio']].values.reshape(V_df.loc[V_df['demora']>0][['inicio']].values.size,)/100.0)
print "Anio promedio de llegada: %s"%np.mean(V_df.loc[V_df['demora']>0][['llegada']].values.reshape(V_df.loc[V_df['demora']>0][['llegada']].values.size,)/100.0)
print "Tiempo medio de demora: %s"%np.mean(12*V_df.loc[V_df['demora']>0][['demora']].values.reshape(V_df.loc[V_df['demora']>0][['demora']].values.size,)/100.0)


##############################################################################
'''
    Segunda parte del procesamiento: Resultados a posteriori
    Cargamos: ASN, ASname, Cuanto demoro en llegar, cuando empezo a subir(>0.3) y cuando llego (=1)
    Tambien agregamos un indice que indica cual fue el mas rapido en llegar
    
    Se descartan los AS cuya demora en llegar es 0.
    
    Se desdobla en analisis en TOTAL y post 2011
'''

c=1
OutputTotal_v=[]

for AS in V_df.loc[V_df['demora']>0].sort('demora',ascending=True)[['ASN']].values:
    name=ASnames_df.loc[ASnames_df['ASN']==AS[0]]
    ASclass=ASclass_df.loc[ASclass_df['as']==AS[0]]
    
    if name['ASname'].values.size>0:
        NAME=name['ASname'].values[0]
    else:
        NAME=''
        
    if ASclass['type'].values.size>0:
        AStype=ASclass['type'].values[0]
    else:
        AStype=''
    
    OutputTotal_v.append((c,V_df.loc[V_df['ASN']==AS[0]]['demora'].values[0],V_df.loc[V_df['ASN']==AS[0]]['llegada'].values[0],AS[0],NAME,AStype))
    c+=1
 
OutputTotal_df=pd.DataFrame(OutputTotal_v,columns=['rank','demora','llegada','ASN','ASname','type'])
   
c=1
Outputpost2011_v=[]

for AS in V_df.loc[(V_df['demora']>0) & (V_df['llegada']>201100)].sort('demora',ascending=True)[['ASN']].values:
    name=ASnames_df.loc[ASnames_df['ASN']==AS[0]]
    ASclass=ASclass_df.loc[ASclass_df['as']==AS[0]]
    
    if name['ASname'].values.size>0:
        NAME=name['ASname'].values[0]
    else:
        NAME=''
        
    if ASclass['type'].values.size>0:
        AStype=ASclass['type'].values[0]
    else:
        AStype=''
    
    Outputpost2011_v.append((c,V_df.loc[V_df['ASN']==AS[0]]['demora'].values[0],AS[0],NAME,AStype))
    c+=1

Outputpost2011_df=pd.DataFrame(Outputpost2011_v,columns=['rank','demora','ASN','ASname','type'])


c=1
Outputpre2011_v=[]

for AS in V_df.loc[(V_df['demora']>0) & (V_df['llegada']<=201100)].sort('demora',ascending=True)[['ASN']].values:
    name=ASnames_df.loc[ASnames_df['ASN']==AS[0]]
    ASclass=ASclass_df.loc[ASclass_df['as']==AS[0]]
    
    if name['ASname'].values.size>0:
        NAME=name['ASname'].values[0]
    else:
        NAME=''
        
    if ASclass['type'].values.size>0:
        AStype=ASclass['type'].values[0]
    else:
        AStype=''
    
    Outputpre2011_v.append((c,V_df.loc[V_df['ASN']==AS[0]]['demora'].values[0],AS[0],NAME,AStype))
    c+=1

Outputpre2011_df=pd.DataFrame(Outputpre2011_v,columns=['rank','demora','ASN','ASname','type'])

##############################################################################
'''
    Resultados producto de hacer el ordenamiento y agregar el campo TYPE
'''

print "ASes that have reached the TOPcore and were there in the past 6 months: %s"%OutputTotal_df['type'].size
print "CPs that have reached the TOPcore and were there in the past 6 months: %s"%OutputTotal_df.loc[OutputTotal_df['type']=='Content']['type'].size
print '----------'
print "ASes that have reached the TOPcore before 2011 and were there in the past 6 months: %s"%Outputpre2011_df['type'].size
print "CPs that have reached the TOPcore before 2011 and were there in the past 6 months: %s"%Outputpre2011_df.loc[Outputpre2011_df['type']=='Content']['type'].size
print '----------'
print "ASes that have reached the TOPcore since 2011 and were there in the past 6 months: %s"%Outputpost2011_df['type'].size
print "CPs that have reached the TOPcore since 2011 and were there in the past 6 months: %s"%Outputpost2011_df.loc[Outputpost2011_df['type']=='Content']['type'].size

OutputTotal_df.to_csv('makeRanking/OutputTotal.csv',header=True,index=False)
Outputpost2011_df.to_csv('makeRanking/Outputpost2011.csv',header=True,index=False)
Outputpre2011_df.to_csv('makeRanking/Outputpre2011.csv',header=True,index=False)

##############################################################################
'''
    TOPcore CPs and Transit by RIR
'''

RIR_v=['ARIN','RIPE','APNIC','LACNIC','AFRINIC']
CPsbyRIR_v=[]
ContentClassificationByRIR_dict={
'ARIN':[],
'RIPE':[],
'APNIC':[],
'LACNIC':[],
'AFRINIC':[]
}

cc2rir_df=pd.read_csv('/Users/estcarisimo/Doctorado/2017/MovimientoCDNs/Geolocation/cc2rir.csv',sep=',')

for name in OutputTotal_df.loc[OutputTotal_df['type']=='Content']['ASname'].values:
    CPsbyRIR_v.append(cc2rir_df.loc[cc2rir_df['cc']==name[-2:]]['rir'].values[0])
    ContentClassificationByRIR_dict[cc2rir_df.loc[cc2rir_df['cc']==name[-2:]]['rir'].values[0]].append(name)

CPsbyRIR_a=np.array(CPsbyRIR_v)
for RIR in RIR_v:
    print RIR,CPsbyRIR_a[CPsbyRIR_a==RIR].size
    
#

TransitbyRIR_v=[]

TransitClassificationByRIR_dict={
'ARIN':[],
'RIPE':[],
'APNIC':[],
'LACNIC':[],
'AFRINIC':[]
}

for name in OutputTotal_df.loc[OutputTotal_df['type']!='Content'][['ASN','ASname']].values:
    if len(name[1])>2:
        if cc2rir_df.loc[cc2rir_df['cc']==name[1][-2:]]['rir'].values.size>0:
            TransitbyRIR_v.append(cc2rir_df.loc[cc2rir_df['cc']==name[1][-2:]]['rir'].values[0])
            TransitClassificationByRIR_dict[cc2rir_df.loc[cc2rir_df['cc']==name[1][-2:]]['rir'].values[0]].append(name[1])
        else:
            print name[0]
    else:
            print name[0]

TransitbyRIR_a=np.array(TransitbyRIR_v)
for RIR in RIR_v:
    print RIR,TransitbyRIR_a[TransitbyRIR_a==RIR].size

##############################################################################
'''
    Seccion de graficos
'''


# Histograma del tiempo necesario para alcanzar el TOPcore por estos ASes que crecen abruptamente

fig, ax1 = pl.subplots(1,figsize=(10, 7))
ax1.hist(12*V_df.loc[V_df['demora']>0][['demora']].values/100.0)
ax1.set_title('Histograma del demora para alcanzar el TOPcore',fontsize=24)   
ax1.set_ylabel('frecuencia',fontsize=20)
ax1.set_xlabel('Demora [mes]',fontsize=20)
fig.subplots_adjust(hspace=0)
fig.tight_layout()
fig.savefig('makeRanking/histYearsToGetTheTop.pdf')

# CDF arribos al TOPcore: CP vs Transit (Overall)

#num_bins = 30
#
#fig, ax1 = pl.subplots(1,figsize=(10, 7))
#
#ax1.yaxis.grid(True, linestyle='-', color='#bababa',alpha=0.5)
#ax1.xaxis.grid(True, linestyle='-', color='#bababa',alpha=0.5)
#
#counts, bin_edges = np.histogram(\
#    (OutputTotal_df.loc[OutputTotal_df['type']=='Content'][['llegada']].values/100.0),\
#    bins=num_bins,\
#    normed=True\
#    )
#cdf = np.cumsum(counts)*(bin_edges[1]-bin_edges[0])
#ax1.step(bin_edges[1:], cdf,linewidth=2,label='CP',alpha=0.8)
#
#intPercentile_CP_a=(np.percentile(cdf,[20,40,60,80,100])*\
#                    OutputTotal_df.loc[OutputTotal_df['type']=='Content'][['llegada']].values.size).astype(int)
#
#counts, bin_edges = np.histogram(\
#    (OutputTotal_df.loc[OutputTotal_df['type']!='Content'][['llegada']].values/100.0),\
#    bins=num_bins,\
#    normed=True\
#    )
#cdf = np.cumsum(counts)*(bin_edges[1]-bin_edges[0])
#ax1.step(bin_edges[1:], cdf,linewidth=2,label='Transit',alpha=0.8)
#
#intPercentile_Transit_a=(np.percentile(cdf,[20,40,60,80,100])*\
#                    OutputTotal_df.loc[OutputTotal_df['type']!='Content'][['llegada']].values.size).astype(int)
#
##ax1.set_yticks(np.arange(0,1.01,0.1),minor=False)
#ticks=np.around(np.arange(0,1.01,0.1), decimals=1)
#ax1.set_yticks(ticks,minor=False)
#ylabels=[]
#for ts in ticks:
#    if ts not in (np.array([20,40,60,80,100])/100.0).tolist():
#        ylabels.append(ts)
#    else:
#        ylabels.append("(%s/%s) %s"%(intPercentile_Transit_a[int(ts*5)-1],intPercentile_CP_a[int(ts*5)-1],ts))
#ax1.set_yticklabels(ylabels, minor=False,ha='right',fontsize=8)
#ax1.set_xticks(np.arange(0,int(bin_edges[1:].max())+1,1).astype(int),minor=False)    
#
#ax1.axis([bin_edges[1:].min(),bin_edges[1:].max(),0,1])
#ax1.set_title('When ASes joined the TOPcore the first time',fontsize=24)   
#ax1.set_ylabel('Fraction of arrivals',fontsize=20)
#ax1.set_xlabel('',fontsize=20)
##ax1.set_xticklabels(ax1.get_xticklabels(), rotation = 45, ha="right")
#pl.xticks(rotation=45, ha="right")
#ax1.tick_params(labelsize=15)
#ax1.legend(loc='upper left',ncol=1,frameon=True,fontsize=17)
#fig.subplots_adjust(hspace=0)
#fig.tight_layout()
#fig.savefig('makeRanking/ECDFYearsLlegada_CPvsTransitOverall.pdf')

ArrivalTime_CP_a=np.sort(OutputTotal_df.loc[OutputTotal_df['type']=='Content'].llegada.values/100.0)
ArrivalTime_Transit_a=np.sort(OutputTotal_df.loc[OutputTotal_df['type']!='Content'].llegada.values/100.0)

qArrivalTime_CP_v=[]
qArrivalTime_Transit_v=[]

for q in (np.floor(np.arange(0,ArrivalTime_CP_a.size,ArrivalTime_CP_a.size/4.0))).astype(int)[1:]:
     qArrivalTime_CP_v.append((ArrivalTime_CP_a[q],q))

for q in (np.floor(np.arange(0,ArrivalTime_Transit_a.size,ArrivalTime_Transit_a.size/4.0))).astype(int)[1:]:
     qArrivalTime_Transit_v.append((ArrivalTime_Transit_a[q],q))

fig, ax1 = pl.subplots(1,figsize=(10, 7))

ax1.yaxis.grid(True, linestyle='-', color='#bababa',alpha=0.5)
ax1.xaxis.grid(True, linestyle='-', color='#bababa',alpha=0.5)


ax1.step(ArrivalTime_CP_a,np.arange(ArrivalTime_CP_a.size)/float(ArrivalTime_CP_a.size),linewidth=2,label='CP (%s)'%ArrivalTime_CP_a.size,color='blue',alpha=0.8)
ax1.step(ArrivalTime_Transit_a,np.arange(ArrivalTime_Transit_a.size)/float(ArrivalTime_Transit_a.size),linewidth=2,color='green',label='Transit (%s)'%ArrivalTime_Transit_a.size,alpha=0.8)

for YYYY,count in qArrivalTime_CP_v:
    ax1.plot([ArrivalTime_Transit_a.min(),YYYY],[count/float(ArrivalTime_CP_a.size),count/float(ArrivalTime_CP_a.size)],linestyle='-.',linewidth=2,color='#5d5d5d')
    ax1.plot([YYYY,YYYY],[0,count/float(ArrivalTime_CP_a.size)],linestyle='-.',linewidth=2,color='#5d5d5d')
for YYYY,count in qArrivalTime_Transit_v:
    ax1.plot([YYYY,YYYY],[0,count/float(ArrivalTime_Transit_a.size)],linestyle='-.',linewidth=2,color='#5d5d5d')

ax1.set_yticks(np.arange(0,1.01,0.1),minor=False)
ax1.set_xticks(np.arange(int(ArrivalTime_Transit_a.min()),int(ArrivalTime_Transit_a.max())+2,1).astype(int),minor=False)    

ax1.axis([ArrivalTime_Transit_a.min(),int(ArrivalTime_Transit_a.max())+1,0,1])
ax1.set_title('Arrival of current TOPcore ASes at the TOPcore',fontsize=24)   
ax1.set_ylabel('Fraction of arrivals',fontsize=20)
ax1.set_xlabel('',fontsize=20)
pl.xticks(rotation=90, ha="center")
ax1.tick_params(labelsize=20)
ax1.legend(loc='upper left',ncol=1,frameon=True,fontsize=17)
fig.subplots_adjust(hspace=0)
fig.tight_layout()

#fig.show()
fig.savefig('makeRanking/ECDFYearsLlegada_CPvsTransitOverall.pdf')

# CDF del tiempo necesario para alcanzar el TOPcore: CP vs Transit (Overall)

num_bins = 30

fig, ax1 = pl.subplots(1,figsize=(10, 7))

ax1.yaxis.grid(True, linestyle='-', color='#bababa',alpha=0.5)
ax1.xaxis.grid(True, linestyle='-', color='#bababa',alpha=0.5)

counts, bin_edges = np.histogram(\
    (12*OutputTotal_df.loc[OutputTotal_df['type']=='Content'][['demora']].values/100.0),\
    bins=num_bins,\
    normed=True\
    )
cdf = np.cumsum(counts)*(bin_edges[1]-bin_edges[0])
ax1.step(bin_edges[1:], cdf,linewidth=2,label='CP',alpha=0.8)

counts, bin_edges = np.histogram(\
    (12*OutputTotal_df.loc[OutputTotal_df['type']!='Content'][['demora']].values/100.0),\
    bins=num_bins,\
    normed=True\
    )
cdf = np.cumsum(counts)*(bin_edges[1]-bin_edges[0])
ax1.step(bin_edges[1:], cdf,linewidth=2,label='Transit',alpha=0.8)

ax1.set_yticks(np.arange(0,1.01,0.1),minor=False)
ax1.set_xticks(np.arange(0,int(bin_edges[1:].max()),12).astype(int),minor=False)    

ax1.axis([bin_edges[1:].min(),bin_edges[1:].max(),0,1])
ax1.set_title('Overall',fontsize=24)   
ax1.set_ylabel('ECDF',fontsize=20)
ax1.set_xlabel('# of months to reach the TOP core',fontsize=20)
ax1.tick_params(labelsize=15)
ax1.legend(loc='lower right',ncol=1,frameon=True,fontsize=17)
fig.subplots_adjust(hspace=0)
fig.tight_layout()
fig.savefig('makeRanking/ECDFYearsToGetTheTop_CPvsTransitOverall.pdf')

# CDF del tiempo necesario para alcanzar el TOPcore: CP vs Transit (2011/Breakpoint)

num_bins = 30

fig, (ax1,ax2) = pl.subplots(2,figsize=(10, 7))

ax1.yaxis.grid(True, linestyle='-', color='#bababa',alpha=0.5)
ax1.xaxis.grid(True, linestyle='-', color='#bababa',alpha=0.5)

counts, bin_edges = np.histogram(\
    (12*Outputpre2011_df.loc[Outputpre2011_df['type']=='Content'][['demora']].values/100.0),\
    bins=num_bins,\
    normed=True\
    )
cdf = np.cumsum(counts)*(bin_edges[1]-bin_edges[0])
ax1.step(bin_edges[1:], cdf,linewidth=2,label='CP',alpha=0.8)

counts, bin_edges = np.histogram(\
    (12*Outputpre2011_df.loc[Outputpre2011_df['type']!='Content'][['demora']].values/100.0),\
    bins=num_bins,\
    normed=True\
    )
cdf = np.cumsum(counts)*(bin_edges[1]-bin_edges[0])
ax1.step(bin_edges[1:], cdf,linewidth=2,label='Transit',alpha=0.8)

ax1.set_yticks(np.arange(0,1.01,0.1),minor=False)
ax1.set_xticks(np.arange(0,int(bin_edges[1:].max()),12).astype(int),minor=False)    

ax1.axis([bin_edges[1:].min(),bin_edges[1:].max(),0,1])
ax1.set_title('Before 2011',fontsize=24)   
ax1.set_ylabel('ECDF',fontsize=20)
ax1.set_xlabel('',fontsize=20)
ax1.tick_params(labelsize=15)
ax1.legend(loc='lower right',ncol=1,frameon=True,fontsize=17)

#

ax2.yaxis.grid(True, linestyle='-', color='#bababa',alpha=0.5)
ax2.xaxis.grid(True, linestyle='-', color='#bababa',alpha=0.5)

counts, bin_edges = np.histogram(\
    (12*Outputpost2011_df.loc[Outputpost2011_df['type']=='Content'][['demora']].values/100.0),\
    bins=num_bins,\
    normed=True\
    )
cdf = np.cumsum(counts)*(bin_edges[1]-bin_edges[0])
ax2.step(bin_edges[1:], cdf,linewidth=2,label='CP',alpha=0.8)

counts, bin_edges = np.histogram(\
    (12*Outputpost2011_df.loc[Outputpost2011_df['type']!='Content'][['demora']].values/100.0),\
    bins=num_bins,\
    normed=True\
    )
cdf = np.cumsum(counts)*(bin_edges[1]-bin_edges[0])
ax2.step(bin_edges[1:], cdf,linewidth=2,label='Transit',alpha=0.8)

ax2.set_yticks(np.arange(0,1.01,0.1),minor=False)
ax2.set_xticks(np.arange(0,int(bin_edges[1:].max()),12).astype(int),minor=False)        

ax2.axis([bin_edges[1:].min(),bin_edges[1:].max(),0,1])

ax2.set_title('After 2011',fontsize=24)   
ax2.set_ylabel('ECDF',fontsize=20)
ax2.set_xlabel('# of months to reach the TOP core',fontsize=20)
ax2.tick_params(labelsize=15)
ax2.legend(loc='lower right',ncol=1,frameon=True,fontsize=17)
fig.subplots_adjust(hspace=0)
fig.tight_layout()
fig.savefig('makeRanking/ECDFYearsToGetTheTop_CPvsTransit2011.pdf')

# Histograma del tiempo de inicio estos ASes que crecen abruptamente

fig, ax1 = pl.subplots(1,figsize=(10, 7))
ax1.hist(V_df.loc[V_df['demora']>0][['inicio']].values/100.0)
ax1.set_title('Histograma del tiempo de inicio',fontsize=24)   
ax1.set_ylabel('frecuencia',fontsize=20)
ax1.set_xlabel('Anio',fontsize=20)
fig.subplots_adjust(hspace=0)
fig.tight_layout()
fig.savefig('makeRanking/histBeginningYear.pdf')

# Histograma del tiempo de arribo estos ASes que crecen abruptamente

fig, ax1 = pl.subplots(1,figsize=(10, 7))
ax1.hist(V_df.loc[V_df['demora']>0][['llegada']].values/100.0)
ax1.set_title('Histograma del tiempo de llegada',fontsize=24)   
ax1.set_ylabel('frecuencia',fontsize=20)
ax1.set_xlabel('Anio',fontsize=20)
fig.subplots_adjust(hspace=0)
fig.tight_layout()
fig.savefig('makeRanking/histReachingYear.pdf')

# Heatmap del tiempo inicio vs la demora

#https://matplotlib.org/examples/pylab_examples/hist2d_log_demo.html

fig, ax1 = pl.subplots(1,figsize=(10, 7))
pl.hist2d(
    12*V_df.loc[V_df['demora']>0][['demora']].values.reshape(V_df.loc[V_df['demora']>0][['demora']].values.size,)/100.0,
    V_df.loc[V_df['demora']>0][['inicio']].values.reshape(V_df.loc[V_df['demora']>0][['inicio']].values.size,)/100.0,
    bins=20,
    norm=LogNorm()
)
pl.colorbar()
fig.subplots_adjust(hspace=0)
fig.tight_layout()
fig.savefig('makeRanking/HeatmapYearstoBeginningYear.pdf')

# Heatmap del tiempo llegada vs la demora
#
#

#fig, ax1 = pl.subplots(1,figsize=(10, 7))
#ax2.yaxis.grid(True, linestyle='-', color='#bababa',alpha=0.5)
#ax2.xaxis.grid(True, linestyle='-', color='#bababa',alpha=0.5)
#HM=pl.hist2d(
#    12*V_df.loc[V_df['demora']>0][['demora']].values.reshape(V_df.loc[V_df['demora']>0][['demora']].values.size,)/100.0,
#    V_df.loc[V_df['demora']>0][['llegada']].values.reshape(V_df.loc[V_df['demora']>0][['llegada']].values.size,)/100.0,
#    bins=20,
#    norm=LogNorm()
#)
#
#cbar=pl.colorbar()
#cbarTICKS=(np.arange(HM[0].min(),HM[0].max()+1,1)).astype(int)
#cbar.ax.set_yticklabels(cbarTICKS)
#cbar.ax.set_ylabel('# of ASes',fontsize=15)
##cbar.ax.set_clim(HM[0].min(),HM[0].max()+1)
#
##ax1.set_title('Histograma del tiempo de llegada',fontsize=24)
#ax1.tick_params(labelsize=15)   
#ax1.set_ylabel('Reaching year',fontsize=20)
#ax1.set_xlabel('# of month to get the TOPcore',fontsize=20)
#fig.subplots_adjust(hspace=0)
#fig.tight_layout()
#fig.show()
##fig.savefig('makeRanking/HeatmapYearstoReachingYear.pdf')

YYYY_total_a=V_df.loc[V_df['demora']>0][['llegada']].values.reshape(V_df.loc[V_df['demora']>0][['llegada']].values.size,)/100
demora_total_a=V_df.loc[V_df['demora']>0][['demora']].values.reshape(V_df.loc[V_df['demora']>0][['demora']].values.size,)

YYYY_a=np.arange(np.floor(YYYY_total_a.min()),int(round(YYYY_total_a.max()))+1)*100
breakpoints_demora_a=np.arange(demora_total_a.min(),demora_total_a.max(),(demora_total_a.max()-demora_total_a.min())/float(YYYY_a.size))

Matrix=np.zeros(shape=(YYYY_a.size-1,YYYY_a.size-1))

for i in range(0,YYYY_a.size-1):
    for j in range(0,breakpoints_demora_a.size-1):
        Matrix[(i,j)]=V_df.loc[(V_df['demora']>breakpoints_demora_a[j]) & (V_df['demora']<=breakpoints_demora_a[j+1]) & (V_df['llegada']>YYYY_a[i]) & (V_df['llegada']<=YYYY_a[i+1])][['demora']].size


Matrix = np.ma.masked_where(Matrix == 0, Matrix)

#Colors = cm.Accent
Colors = cm.get_cmap('PuBu',Matrix.max()-1)
Colors.set_bad(color='white')


fig, ax1 = pl.subplots(1,figsize=(10, 7))

im = ax1.pcolor(\
                Matrix,\
                norm=pl.Normalize(vmin=Matrix[np.isfinite(Matrix)].min(),\
                vmax=Matrix[np.isfinite(Matrix)].max()),\
                cmap=Colors,\
                edgecolors='#bababa',\
                linewidths=0.5)

cbar = pl.colorbar(im, ticks=np.arange(0.5,Matrix[np.isfinite(Matrix)].max(),1),ax=ax1)               
cbarTICKS=(np.arange(Matrix.min(),Matrix.max()+1,1)).astype(int)
cbar.ax.set_yticklabels(cbarTICKS)
cbar.ax.set_ylabel('# of ASes',fontsize=15)
#cbar.ax.set_clim(HM[0].min(),HM[0].max()+1)



ax1.set_xticks(np.arange(0,YYYY_a.size))
ax1.set_xticklabels((12*breakpoints_demora_a/100.0).astype(int),rotation=45, minor=False,ha='right',fontsize=8)

ax1.set_yticks(np.arange(0,YYYY_a.size))
ax1.set_yticklabels((YYYY_a/100.0).astype(int), minor=False,fontsize=8)

ax1.axis([0,YYYY_a.size-1,0,YYYY_a.size-1])

ax1.yaxis.grid(True, linestyle='-', color='#bababa',alpha=0.5)
ax1.xaxis.grid(True, linestyle='-', color='#bababa',alpha=0.5)

ax1.tick_params(labelsize=15)   
ax1.set_ylabel('Reaching year',fontsize=20)
ax1.set_xlabel('# of month to get the TOPcore',fontsize=20)
#ax1.set_title('Histograma del tiempo de llegada',fontsize=24)

fig.subplots_adjust(hspace=0)
fig.tight_layout()
fig.show()


#fig.show()
fig.savefig('makeRanking/HeatmapYearstoReachingYear.pdf')