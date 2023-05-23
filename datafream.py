import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler #归一化
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
path = r'D:\BaiduSyncdisk\machine learning test\task1\sample'
data=load_data(path)

data['label'] = data.apply(lambda x: x['label']+x['value1']*1024 +  x['value2']*2048, axis=1) #更新标签
norma_data=is_normalization(data) #归一化
norma_data.fillna(0,inplace=True) #缺失值填补0
norma_data=norma_data.drop(index=data[data['block_num']==1 ].index) #去除单个信息点样例
all_y=norma_data['label'].values
cols = [i for i in data.columns.tolist() if i not in all_col]  # 要去除的列
all_x= norma_data.drop(labels=cols, axis=1)  # 去除非特征列

def model_KNN(all_x,all_y):
    RANDOM_SEED = 16305  # 固定随机种子
    x_train,x_test,y_train,y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=RANDOM_SEED)
    clf = KNeighborsClassifier(n_neighbors=3, weights='distance', p=1)
    clf.fit(x_train, y_train)
    res = clf.score(x_test, y_test)
    print("Accurancy of KNN:", res)

    predict=[]
    predict=clf.predict(all_x)
    ori=all_y.tolist()
    error=[]
    for ind in range(len(ori)):
        if (ori-predict)[ind] !=0 :
            error.append(ind)
    print("error sample is:")
    print(norma_data.loc[error])

def model_Random_forest(all_x,all_y):
    RANDOM_SEED = 4351  # 固定随机种子
    x_train,x_test,y_train,y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=RANDOM_SEED)
    rfc = RandomForestClassifier(n_estimators=21,random_state=0)
    rfc = rfc.fit(x_train,y_train)
    score_r = rfc.score(x_test,y_test)
    print("Accurancy of Random Forest:",score_r)

    predict=[]
    predict=rfc.predict(all_x)
    ori=all_y.tolist()
    error=[]
    for ind in range(len(ori)):
        if (ori-predict)[ind] !=0 :
            error.append(ind)
    print("error sample is:")
    print(norma_data.loc[error])

#model_KNN(all_x,all_y)
model_Random_forest(all_x,all_y)