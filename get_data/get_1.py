import os,re
import pandas as pd
import xlrd,openpyxl
import numpy as np

pd.set_option('chained_assignment', None)

def date(para):
    if type(para) == int:
        delta = pd.Timedelta(str(int(para))+'days')
        time = pd.to_datetime('1899-12-30') + delta
        return time
    else:
        return para


aa,bb='2-7//大峡谷隧道出口右线(K87-82).xls'.split('//')

path=os.path.abspath(os.path.dirname(os.getcwd()))
input=os.path.join(path,'data_2\\大峡谷隧道监测数据\\%s'%(aa))
output=os.path.join(path,'result\\cess_1')
file=os.path.join(input,bb)

df=pd.read_excel(file,None,skiprows=4)
print(df.keys())
apps = []

tag_list=[]
if '%s.csv'%(bb.split('.')[0]) in os.listdir(output):
    os.remove(os.path.join(output,'%s.csv'%(bb.split('.')[0])))

for tag,df in df.items():
    #左线换右线这里一定记得修改！！！
    #左线：if tag[:2]=='ZK' or 'zk':
    #右线：if (tag[0]=='K')|(tag[0]=='k'):
    if (tag[0]=='K')|(tag[0]=='k')|(tag[:2]=='ZK')|(tag[0]=='zk'):
        tag_list.append(tag)
        #print(tag,df)
        a,b=re.findall('([0-9]*)\+[ ]*([0-9]*)',tag)[0]
        distance=int(a)*1000+int(b)
        sheet_name=[i for i in df.columns if '累计' in i]
        df1=df[sheet_name]

        try:
            df1.columns=['G1','G2','G3','AB','BC','AC']
        except:
            continue
        df1.loc[:,'date']=df.iloc[:,0]
        df1['date']=np.array(df1['date'].apply(date))
        df1=df1.dropna(axis = 0)


        if df1.empty:
            continue
        df1['date']=pd.to_datetime((df1['date']))
        df1=df1.set_index(df1['date'])

        df1.loc[:,'date2']=[df1.index[0]]+list(df1.index)[:-1]

        df1['days'] = (df1['date'] - df1['date2']).dt.days
        df1=df1.drop(['date','date2'],axis=1)

        try:
            xx=list(df1['days']).index([i for i in df1['days'] if i>2][0])
            df2=df1.iloc[:xx,:]
        except:
            df2=df1

        df2 = df2.drop(['days'], axis=1)

        df2['BC']=df2['BC'].astype(float)
        df2=df2.resample('D').mean().interpolate() # 填补时间序列空白并取插值

        df2.loc[:, 'tag'] = [tag]*len(df2)
        df2.loc[:, 'dis'] = [distance]*len(df2)

        apps.append(str(len(df2)))
        if '%s.csv'%(bb.split('.')[0]) not in os.listdir(output):
            df2.to_csv(os.path.join(output,'%s.csv'%(bb.split('.')[0])), mode='a')
        else:
            df2.to_csv(os.path.join(output, '%s.csv'%(bb.split('.')[0])), mode='a',header=None)

print(tag_list)
#tag里记录着每个断面的时间长度
with open(os.path.join(output,'tag','%s_tag.txt'%(bb.split('.')[0])),'w')as f:
    f.write('\n'.join(apps))
