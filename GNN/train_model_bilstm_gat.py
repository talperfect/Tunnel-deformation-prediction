# -*- coding:utf-8 -*-
# author: tiger
# datetime:2023/9/7 9:38
# software: PyCharm
"""权重融合"""
import pandas as pd
import numpy as np
import torch,os,time
from bi_lstm_gat import MM
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from itertools import chain
import matplotlib.pyplot as plt

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_datafile(name,random_seed=0,bili=0.7):
    path = os.path.abspath(os.path.dirname(os.getcwd()))
    path_re = os.path.join(path, 'result', 'cess_3')
    df = pd.read_csv(os.path.join(path_re, 'inputdata_%s.csv' % name), header=None, encoding='utf-8-sig')

    data = np.array(df).reshape((-1, 5, 3, 6))

    df1 = pd.read_csv(os.path.join(path_re, 'testdata_%s.csv' % name), header=None, encoding='utf-8-sig')
    label = np.array(df1)

    df2 = pd.read_csv(os.path.join(path_re, 'adj_%s.csv' % name), header=None, encoding='utf-8-sig')
    adj = np.array(df2).reshape(-1, 3, 3)

    data = torch.Tensor(data)
    labels = torch.Tensor(label)
    adj = torch.Tensor(adj)

    data = data.to(device)
    labels = labels.to(device)
    adj = adj.to(device)

    # 创建一个TensorDataset，将数据和标签组合在一起
    dataset = TensorDataset(data, labels, adj)

    # 定义训练集和测试集的比例
    train_size = int(bili * len(dataset))
    test_size = len(dataset) - train_size
    print('训练集共%d条，测试集共%d条'%(train_size,test_size))
    # 使用random_split函数将数据集分成训练集和测试集
    torch.manual_seed(random_seed)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
	

	
    return train_dataset, test_dataset

def train(model, train_loader, criterion, optimizer, num_epochs,output_path):
    adj=pd.read_csv('adj.csv',header=None)
    adj=torch.from_numpy(adj.values).to(device)

    loss_epoch = []


    if not os.path.exists(os.path.join(output_path,'model_jilu')):
        os.makedirs(os.path.join(output_path,'model_jilu'))

    for epoch in range(num_epochs):
        loss_batch = 0
        for batch_idx, (data, labels,a) in enumerate(train_loader):

            outputs, state = model(torch.squeeze(data),torch.squeeze(adj))

            loss = criterion(torch.squeeze(outputs), torch.squeeze(labels))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #loss_batch = (loss_batch*batch_idx + loss.item())/(batch_idx+1)
            loss_batch += loss.item()
            '''
            if (batch_idx + 1) % log_interval == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item()}")
            '''
        loss_batch=loss_batch/len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}],Loss: {loss_batch}')
        loss_epoch.append(loss_batch)


        if not epoch%10:
            torch.save(model.state_dict(), os.path.join(output_path, 'model_jilu', 'MM_model_weights_%d.pth'%(int(epoch/10))))

        if epoch==num_epochs-1:
            torch.save(model.state_dict(), os.path.join(output_path, 'MM_model_weights.pth'))


        '''
        # 打印模型的参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
        '''

    pd.DataFrame(loss_epoch).to_csv(os.path.join(output_path,'loss_epoch.csv'),index=False,header=None)


# 步骤 6：保存模型(根据需要)

# 主训练循环
if __name__ == "__main__":
    # 在训练之前记录开始时间
    start_time = time.time()
    print(device)
    name = '左线'
    batch_size = 1

    # 断面数
    num_tunnel = 3

    # 定义模型
    model = MM(5).to(device)

    # 步骤 3：定义损失函数和优化器
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.0001)
    #optimizer=optimizer.to(device)
	
    train_dataset, test_dataset=read_datafile(name,0)
    print(train_dataset[0])
	
    #print(train_dataset)
    '''
    # 加载数据
    train_dataset = CustomDataset(name)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    '''
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    num_epochs = 500
    log_interval = 50

    path = os.path.abspath(os.path.dirname(os.getcwd()))

    train(model, train_loader, criterion, optimizer, num_epochs,os.path.join(path, 'model_base', '3.30trasj'))

    # 在训练结束后记录结束时间
    end_time = time.time()

    # 计算训练时间
    training_time = end_time - start_time
    print("模型训练时间：", training_time, "秒")
