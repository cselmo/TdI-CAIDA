import pandas as pd
import numpy as np
import os

def select_as(asn_df,core_df,from_date,from_th, to_date, to_th, maximum_months):
    core_tmp=(core_df.head(from_date)<to_th).all(axis=0)
    sel_cols=np.array(core_tmp.index.tolist())[core_tmp.tolist()]
    core_tmp=(core_df.loc[from_date-1:to_date-1,sel_cols]>to_th).any(axis=0)
    sel_cols=np.array(core_tmp.index.tolist())[core_tmp.tolist()] 
    core_tmp=(core_df.loc[from_date-1:to_date-1,sel_cols]<from_th).any(axis=0)
    sel_cols=np.array(core_tmp.index.tolist())[core_tmp.tolist()]
    out_df=pd.DataFrame(columns=["ASNumber", "ShortName", "Country", "StartGrow", "StopGrow", "MonthGrow"])
    for col in sel_cols:
        core_tmp=(core_df.loc[from_date-1:to_date-1,col])>to_th
        sel_dates=np.array(core_tmp.index.tolist())[core_tmp.tolist()]
        min_to_th_month=int(sel_dates[0])
        core_tmp=(core_df.loc[from_date-1:min_to_th_month,col])<from_th
        sel_dates=[]
        sel_dates=sel_dates+list(np.array(core_tmp.index.tolist())[core_tmp.tolist()])
        if (len(sel_dates)!=0):
            max_from_th_month=int(sel_dates[-1])
            if (max_from_th_month<min_to_th_month and min_to_th_month-max_from_th_month<=maximum_months):
                try:
                    asn_row=asn_df.loc["AS"+col].to_dict()
                except:
                    asn["ShortName"]="Unknown"
                    asn["Country"]="Unknown"
                asn_row["ASNumber"]="AS"+col
                asn_row["StartGrow"]=str(int(max_from_th_month/12+1998)) + " / " + str(max_from_th_month%12 + 1)
                asn_row["StopGrow"]=str(int(min_to_th_month/12+1998)) + " / " + str(min_to_th_month%12 + 1)
                asn_row["MonthGrow"]=str(min_to_th_month-max_from_th_month)
                out_df=out_df.append(asn_row,ignore_index=True)
    return out_df

if __name__=="__main__":
    select_as(3,4,5,6,7)

