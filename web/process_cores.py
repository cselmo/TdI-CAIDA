import pandas as pd
import numpy as np
df=pd.read_csv("data/cores_norm.csv", index_col=None)
#remap=pd.read_csv("remap.txt",sep=" ",dtype={'old_node':str,'new_node':str})
#rename={}
#for index in remap.index:
#    old_node=remap[remap.index==index]["old_node"].values[0]
#    new_node=remap[remap.index==index]["new_node"].values[0]
#    if(int(old_node)>1000000):
#        df.drop(new_node,1,inplace=True)
#        print(index)
#        print(old_node)
#    else:
#        rename[new_node]=int(old_node)
#    df.rename(columns=rename,inplace=True)
for col in df.columns:
    if col!="time" and col!="top_core":    
        df[col]=df[col]/df["top_core"]
        print(col)

df.drop("top_core",1, inplace=True)
df.to_csv("data/cores_norm2.csv")
