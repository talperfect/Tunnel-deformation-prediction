import os,re
import pandas as pd
import xlrd,openpyxl

path=os.path.abspath(os.path.dirname(os.getcwd()))

input=os.path.join(path,'result\\cess_1')
name=[i for i in os.listdir(input) if i[-3:]=='csv' and '左' in i and i!='左线.csv']
print(name)
dfs=[pd.read_csv(os.path.join(input,i)) for i in name]
print(dfs)
df=pd.concat(dfs,ignore_index=True)
print(df.columns)
df = df.sort_values(by=['dis', 'date'])
df.to_csv(os.path.join(input,'左线.csv'),index=False,encoding='utf_8_sig')