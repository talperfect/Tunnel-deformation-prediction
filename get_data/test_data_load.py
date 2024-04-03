import os,re,datetime
import pandas as pd
import xlrd,openpyxl

def get_time_ss(df):
    tt=df.iloc[-1,0]
    df_ree=df[df[0]==tt]
    aa=int(len(df_ree)/2)
    ss=df_ree.iloc[aa,-2]
    one_day = datetime.timedelta(days=1)
    date = pd.to_datetime(tt)
    next_day = date + one_day
    return next_day,ss,df_ree[df_ree[7]==ss]

name='右线.csv'

path=os.path.abspath(os.path.dirname(os.getcwd()))
input=os.path.join(path,'result\\cess_1')
output=os.path.join(path,'result\\cess_2')
file=os.path.join(input,name)

df=pd.read_csv(file)
df1=pd.read_excel(os.path.join(output,'inputdata_'+name[:-4]+'.xlsx'),None,header=None)
#print(df)
df['date']=pd.to_datetime((df['date']))

df_res=pd.DataFrame()

for i,j in df1.items():
    tt,ss,df_b=get_time_ss(j)
    print(tt,ss)
    dfif=df[(df['date']==tt)&(df['tag']==ss)]
    if dfif.empty:
        df_b.columns=df.columns
        dfif=df_b
        dfif.iloc[0,0]=tt
        df_res=pd.concat([df_res,dfif])
    else:
        df_res=pd.concat([df_res,dfif])
df_res.to_csv(os.path.join(output,'testdata_'+name[:-4]+'.csv'),header=None,index=False,encoding='utf_8_sig')










