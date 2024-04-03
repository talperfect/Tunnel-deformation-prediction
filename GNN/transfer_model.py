import pandas as pd
import numpy as np
import torch,os
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from itertools import chain
import matplotlib.pyplot as plt
import joblib,time
from bi_lstm_gat import MM as bilstmgat
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from train_model_bilstm_gat import read_datafile,train

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 在训练之前记录开始时间
start_time = time.time()

path = os.path.abspath(os.path.dirname(os.getcwd()))

# 假设你的模型是一个示例模型
name = '右线'
batch_size = 1

# 断面数
num_tunnel = 3
model = bilstmgat(5).to(device)

# 加载模型参数

model.load_state_dict(torch.load(os.path.join(path, 'model_base', '1.20model', 'MM_model_weights.pth')))

# 冻结 gcn 部分参数
for param in model.gcns.parameters():
    param.requires_grad = False

for param in model.cov1.parameters():
    param.requires_grad = False

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

train_dataset, test_dataset = read_datafile(name, bili=0.3)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(device)

num_epochs = 500
log_interval = 50
path = os.path.abspath(os.path.dirname(os.getcwd()))
train(model, train_loader, criterion, optimizer, num_epochs,os.path.join(path, 'model_base', '3.11model_tran'))

# 在训练结束后记录结束时间
end_time = time.time()

# 计算训练时间
training_time = end_time - start_time
print("模型训练时间：", training_time, "秒")
