import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler #归一化
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter(log_dir='test')

pd.set_option('display.width', 150)
pd.set_option('display.max_columns', 10)
# pd.set_option('display.max_rows', 10)

f_col=['label','feature11','feature12','feature13','feature14','feature21','feature22','feature23','feature24','feature31','feature32','feature33','feature34','feature41','feature42','feature43','feature44','feature51','feature52','feature53','feature54','feature61','feature62','feature63','feature64','feature71','feature72','feature73','feature74','feature81','feature82','feature83','feature84','feature91','feature92','feature93','feature94','feature101','feature102','feature103','feature104']
g_col=['value1','g_feature1','g_feature2','g_feature3','g_feature4','value2','g_feature5','g_feature6','g_feature7','g_feature8']
c_col=['feature11','feature12','feature13','feature14','feature21','feature22','feature23','feature24','feature31','feature32','feature33','feature34','feature41','feature42','feature43','feature44','feature51','feature52','feature53','feature54','feature61','feature62','feature63','feature64','feature71','feature72','feature73','feature74','feature81','feature82','feature83','feature84','feature91','feature92','feature93','feature94','feature101','feature102','feature103','feature104']
g_col_n=['g_feature1','g_feature2','g_feature3','g_feature4','g_feature5','g_feature6','g_feature7','g_feature8']
all_col=['feature11','feature12','feature13','feature14','feature21','feature22','feature23','feature24','feature31','feature32','feature33','feature34','feature41','feature42','feature43','feature44','feature51','feature52','feature53','feature54','feature61','feature62','feature63','feature64','feature71','feature72','feature73','feature74','feature81','feature82','feature83','feature84','feature91','feature92','feature93','feature94','feature101','feature102','feature103','feature104','g_feature1','g_feature2','g_feature3','g_feature4','g_feature5','g_feature6','g_feature7','g_feature8']
g_sample_num=[]
def load_data(path):
    ori=[]
    gfeature=[]
    sample_line_num_l = []  # 获取填图块个数
    is_normalization=[]
    doc=[]
    line_num=[]
    samples = os.listdir(path)  # 函数返回值是一个列表，其中各元素均为字符串，分别是各路径名和文件名。
    for i in range(len(samples)):
        line_data = []
        sample_path = path + '/' + samples[i]
        f = open(sample_path, encoding='ansi')
        lines = f.readlines()
        for line in lines:
            if line.isspace() or line.startswith('#'):  # 如果读到空行或者#号开头，就跳过
                continue
            else:
                line = line.replace("\n", "")  # 去除文本中的换行等等，可以追加其他操作
                line = line.replace("\t", "")
                line_data.append(line)  # 处理完成后的行，追加到列表中
        cur_sample_num = int(line_data[0])# 保存样例个数
        g_sample_num.append(cur_sample_num)
        cur_is_normalization=int(line_data[2])  # 保存是否归一化行
        cur_g_feature = line_data[3] + ' ' + line_data[4]
        cur_g_feature = list(map(float, cur_g_feature.split()))
        for j in range(5):
            line_data.pop(0)  # 删除样例个数行
        assert cur_sample_num == len(line_data)  # 判断样例个数行数值是否和样例个数相同
        for new_line in line_data:
            line_num.append(line_data.index(new_line)+1)
            sample_line_num_l.append(int((len(new_line.split()) - 1) / 4))
            new_line = list(map(float, new_line.split()))  # 转化浮点数
            ori.append(new_line)
            gfeature.append(cur_g_feature)
            is_normalization.append(cur_is_normalization)
            doc.append(i+1)
    data =pd.DataFrame(ori,columns=f_col)
    g_feature=pd.DataFrame(gfeature,columns=g_col)
    data=pd.concat([data, g_feature], axis=1)
    data['block_num'] = pd.Series(sample_line_num_l)
    data['is_normalization']=pd.Series(is_normalization)
    data['doc'] = pd.Series(doc)
    data['line_num']=pd.Series(line_num)
    return data

def process_min_max(data):
    scaler = MinMaxScaler()  # 实例化
    scaler = scaler.fit(data)  # fit，在这里本质是生成min(x)和max(x)
    result = scaler.transform(data)
    return result

def is_normalization(data):
    cur_colum = data.columns.tolist()
    all_x=pd.DataFrame(columns=cur_colum)
    s_col = c_col.copy()
    s_col.insert(0, 'label')
    for i in range(1,data['doc'].max()+1): #对每一个文本文件进行
        cur_process=data[data['doc']==i ]
        cur_index = cur_process.index
        if cur_process.loc[i*(g_sample_num[i-1])-1]['is_normalization']==1 :
            all_x = pd.concat([all_x, cur_process])
            continue
        cols=[i for i in cur_colum if i not in c_col] #要去除的列
        process=cur_process.drop(labels=cols, axis=1) #去除非特征列
        process=pd.DataFrame(process_min_max(process),columns=c_col,index=cur_index) #对特征列归一化并还原回datafream
        process=pd.concat([cur_process['label'],process],axis=1)
        process=pd.concat([process,cur_process.drop(labels=s_col, axis=1)],axis=1)
        all_x=pd.concat([all_x,process])
    return all_x #返回归一化的特征向量 40维 不含全局特征
path = r'H:\BaiduSyncdisk\machine learning test\task1\sample'
data=load_data(path)

data['label'] = data.apply(lambda x: x['label']+x['value1']*1024 +  x['value2']*2048, axis=1) #更新标签
norma_data=is_normalization(data) #归一化
norma_data.fillna(0,inplace=True) #缺失值填补0
norma_data=norma_data.drop(index=data[data['block_num']==1 ].index) #去除单个信息点样例
all_y=norma_data['label'].values
cols = [i for i in data.columns.tolist() if i not in all_col]  # 要去除的列
all_x= norma_data.drop(labels=cols, axis=1)  # 去除非特征列


all_data=torch.from_numpy(all_x.values)
all_data = all_data.to(torch.float)
all_target=torch.tensor(all_y)
all_target=all_target.to(torch.long)
dataset=Data.TensorDataset(all_data,all_target)

train_data,test_data=random_split(dataset,[round(0.8*all_data.shape[0]),round(0.2*all_target.shape[0])],generator=torch.Generator().manual_seed(42))


# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 超参数设置
input_size =48
hidden1_size = 256
hidden2_size = 256
num_classes = 4096
num_epochs = 16
batch_size = 32
learning_rate = 0.001

device = torch.device("cuda")#使用GPU

# 加载数据集
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


# 初始化模型、损失函数和优化器
model = MLP(input_size, hidden1_size, hidden2_size, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
samples=0
correct=0
# 训练模型
for epoch in range(num_epochs):
    for i, (input_t, labels_t) in enumerate(train_loader):
        #input_t= input_t.reshape(-1, input_size)
        # 前向传播
        input_t = input_t.to(device)
        labels_t = labels_t.to(device)
        outputs = model(input_t)
        loss = criterion(outputs, labels_t)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        yhat = torch.max(outputs, 1)[1]
        correct += torch.sum(yhat == labels_t)
        samples += input_t.shape[0]
        if (i+1) % 40== 0:
            #print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            print("Epoch{}:[{}/{}({:.0f}%)] Loss:{:.6f},Accuarcy:{:.3f}".format(epoch + 1
                                                                                , samples
                                                                                , num_epochs * len(train_loader.dataset)
                                                                                , 100 * samples / (num_epochs * len(train_loader.dataset))
                                                                                , loss.data.item()
                                                                                , float(100 * correct / samples)))  # 分子代表：已经查看过的数据有多少，分母代表：在现有的epochs数据下，模型一共需要查看多少数据
    writer.add_scalar('loss1', loss, epoch)
# 测试模型
with torch.no_grad():
    correct = 0
    total = 0
    for input, labels in test_loader:
      #  input = input.reshape(-1, input_size)
        input = input.to(device)
        labels = labels.to(device)
        outputs = model(input)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'Accuracy of the model is: { correct / total}')
