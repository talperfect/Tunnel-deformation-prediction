import os,re
import pandas as pd
import xlrd,openpyxl
pd.options.mode.chained_assignment = None  # 禁用警告

name='左线.csv'

path=os.path.abspath(os.path.dirname(os.getcwd()))
input=os.path.join(path,'result\\cess_1')
output=os.path.join(path,'result\\cess_2')
file=os.path.join(input,name)

df=pd.read_csv(file)
#print(df)
#差分函数，计算每列上行与下行的差值：df.diff()
#df['sep']=df['dis'].diff().fillna(0)

#================================================================================
date_num=df.groupby('date').size()
print(date_num.value_counts())
print('total:%d'%len(date_num))
print('可以看出在同一天的断面数量有限，这里取三个为宜')
print('='*40)

date_num=df.groupby('tag').size()#查看各元素个数
print(date_num.value_counts())#统计个数
print('total:%d'%len(date_num))
print('可以看出同一断面数据量很多，考虑要保证数据体量，这里取五个为宜，四个为训练数据，一个为结果')
print('='*40)
#================================================================================
print(len(df))
# 计算均值和标准差
mean = df[['G1','G2','G3','AB','BC','AC']].mean()
std = df[['G1','G2','G3','AB','BC','AC']].std()

# 计算上下限阈值
lower_limit = mean - 3 * std
upper_limit = mean + 3 * std

#print(lower_limit,upper_limit)

# 标记离群值
df = df[(df['BC'] >= lower_limit['BC']) & (df['BC'] <= upper_limit['BC'])]
df = df[(df['G2'] >= lower_limit['G2']) & (df['G2'] <= upper_limit['G2'])]
print(len(df))
df.to_csv(os.path.join(input,'剔除后.csv'),index=False,encoding='utf_8_sig')

#剔除量小数据
c1=df.groupby('date').size()
df1=df.loc[df['date'].isin(c1.index[c1>=3])]

df1.loc[:,'date']=pd.to_datetime((df1['date']))

print(len(df1))
c2=df.groupby('tag').size()

df1=df1.loc[df1['tag'].isin(c2.index[c2>=5])]
print(len(df1))
#================================================================================
print('='*40)
#print(df1)
#最终选择三个断面，5个时间步
#df111=df1.sort_values(by=['date','dis'])#按日期增序
df1=df1.sort_values(by=['dis','date'])

#print(list(df1['date']))

tt = 5
nn = 3
max_dis = 50 #极限距离，两个断面里程差的最大阈值，超过这个值不算临近
rng = pd.date_range(df1['date'].min(),df1['date'].max(),freq ='D')
#print(rng)

writer = pd.ExcelWriter(os.path.join(output, 'inputdata_'+name[:-4]+'.xlsx'))
n=0
for i in range(len(rng)-tt+1):
    timeindex=pd.Series(rng[i:i+tt])

    #保证日期连续
    if False not in list(timeindex.isin(df1['date'])):
        co=df1.loc[df1['date'].isin(timeindex)]#该3天内所有断面数据

        df_dis_cha = co['dis'].diff()#计算距离差值
        df_dis_cha.fillna(0, inplace=True)#第一行空数据补0


        if df_dis_cha.max()>max_dis:
            lli=df_dis_cha.to_list()
            indices =[0]+ [i for i, value in enumerate(lli) if value >= max_dis]+[len(lli)]
            df_dis_cut=[co.iloc[indices[i]:indices[i+1],:] for i in range(len(indices)-1)]
        else:
            df_dis_cut=[co]

        for co in df_dis_cut:
            a1 = co.groupby('date').size()  # 每天的断面个数

            #保证每个日期三个断面
            if a1.min()>=nn:
                #tags=co.loc[co['date']==a1.index[a1==a1.min()][0],'tag']#最少断面日期的断面tag集合

                vc_co=co['tag'].value_counts()#所有断面存在的天数
                tags_1=vc_co.index[vc_co == tt]#筛选出合格的断面数据，存在5天即为合格

                #5天都出现的断面为合格断面，要保证至少3个
                if len(tags_1)>=nn:
                    print(pd.Series(tags_1))
                    print('=' * 5)
                    for mm in range(len(tags_1)-nn+1):
                        df_re=co[co['tag'].isin(tags_1[mm:mm+nn])]
                        df_re.to_excel(excel_writer=writer, sheet_name='%d'%n,index=False, header=None)
                        n+=1

writer.close()

<<<<<<< HEAD
#df1.to_csv(os.path.join(output, 'see.csv'))
=======
#df1.to_csv(os.path.join(output, 'see.csv'))
>>>>>>> parent of 0e96305 (Delete get_data/input_data_load.py)
