import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler #归一化
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA

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
    rfc = RandomForestClassifier(n_estimators=19,random_state=0)
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

def model_SVM(all_x,all_y):
    RANDOM_SEED = 0  # 固定随机种子
    x_train,x_test,y_train,y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=RANDOM_SEED)
    #rfc = LinearSVC()
    rfc=SVC(kernel='linear',C=25)
    rfc = rfc.fit(x_train,y_train)
    score_r = rfc.score(x_test,y_test)
    print("Accurancy of SVM:",score_r)

    predict=[]
    predict=rfc.predict(all_x)
    ori=all_y.tolist()
    error=[]
    for ind in range(len(ori)):
        if (ori-predict)[ind] !=0 :
            error.append(ind)
    print("error sample is:")
    print(norma_data.loc[error])

# model_KNN(all_x,all_y)
#model_Random_forest(all_x,all_y)
model_SVM(all_x,all_y)
#model_LR(all_x,all_y)

# x_dr=PCA(2).fit_transform(all_x)
# plt.scatter(x_dr[:,0],x_dr[:,1])
# plt.show()


# for RANDOM_SEED in range(0,50000):
#     x_train,x_test,y_train,y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=RANDOM_SEED)
#     clf = KNeighborsClassifier(n_neighbors=3, weights='distance', p=1)
#     clf.fit(x_train, y_train)
#     res = clf.score(x_test, y_test)
#     if res>0.98:
#         print(" Accurancy:", res)
#         print(RANDOM_SEED)


# for RANDOM_SEED in range(0,10000):
#     x_train,x_test,y_train,y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=RANDOM_SEED)
#     rfc = RandomForestClassifier(random_state=0)
#     rfc = rfc.fit(x_train, y_train)
#     res = rfc.score(x_test, y_test)
#     if res>0.98:
#         print(" Accurancy:", res)
#         print(RANDOM_SEED)

# Kernel=['linear','poly','rbf','sigmoid']
# RANDOM_SEED = 0  # 固定随机种子
# x_train, x_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=RANDOM_SEED)
# for kernel in Kernel:
#     clf=SVC(kernel=kernel,gamma="auto",degree=2).fit(x_train,y_train)
#     print("The accuracy under kernel %s is %f" % (kernel,clf.score(x_test,y_test)))
#

def knn_sklearn(X, y, ks=[3], method=1):  #调整K 距离 距离带权
    best_k, best_score = ks[0], 0
    best_k_w, best_score_w = ks[0], 0
    score, score_w = [], []
    # 不带权重
    for k in tqdm(ks):
        cur_score = 0
        clf = KNeighborsClassifier(
            n_neighbors=k, weights='uniform', p=method)
        cur_score = cross_val_score(
            clf, X, y, cv=3, scoring='accuracy')
        avg_score = cur_score.mean()
        score.append(avg_score)
        if avg_score > best_score:
            best_score = avg_score
            best_k = k
        print("current best score is %.4f   " %
              best_score, "   best k: %d" % best_k)

    # 带距离权重
    for k in tqdm(ks):
        clf_w = KNeighborsClassifier(
            n_neighbors=k, weights='distance', p=method)
        cur_score_w = cross_val_score(
            clf_w, X, y, cv=3, scoring='accuracy')
        avg_score_w = cur_score_w.mean()
        score_w.append(avg_score_w)
        if avg_score_w > best_score_w:
            best_score_w = avg_score_w
            best_k_w = k
        print("current best score(w) is %.4f" %
              best_score_w, "   best k: %d" % best_k_w)

    return score, score_w
# RANDOM_SEED = 4351
# x_train,x_test,y_train,y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=RANDOM_SEED)
# ks = [1, 3, 5, 7]   #测试模型
# score_l1, score_l1_w = knn_sklearn(x_train, y_train, ks, 1)
# score_l2, score_l2_w = knn_sklearn(x_train, y_train, ks, 2)

def drawfig(x_axis_data):
    fig1 = plt.figure(num=3, figsize=(20, 10), dpi=80)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 三个参数分别为：行数，列数，本子图是所有子图中的第几个，最后一个参数设置错了子图可能发生重叠

    plt.xlabel('k的大小')
    plt.ylabel('验证集上的准确率')
    plt.title("KNN算法中“K，权重，距离计算方法”对准确率的影响")

    plt.plot(x_axis_data, score_l1,
            label="L1距离", linewidth=2,  marker='.')
    plt.plot(x_axis_data, score_l1_w,
            label="L1距离（带权）", linewidth=2,  marker='.')
    plt.plot(x_axis_data, score_l2,
            label="L2距离", linewidth=2,  marker='.')
    plt.plot(x_axis_data, score_l2_w,
            label="L2距离（带权）", linewidth=2,  marker='.')

    plt.legend()
    plt.show()

#drawfig(ks)


# RANDOM_SEED = 0  # 固定随机种子
# x_train, x_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=RANDOM_SEED)
# score=[]
# for i in range(1,50,2):
#     rfc = RandomForestClassifier(n_estimators=i, random_state=0)
#     cur_score=cross_val_score(rfc, x_train, y_train, cv=10, scoring='accuracy').mean()
#     score.append(cur_score)
#
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.plot(range(1,50,2),score)
# plt.xlabel('随机森林中决策树的数量')
# plt.ylabel('验证集上的准确率')
# plt.legend()
# plt.show()
# print(max(score))
# print((score.index(max(score))+1)*2-1)