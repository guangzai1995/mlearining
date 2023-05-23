import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split

data = []
label = []
sample_num=[]
feature_num=[]
is_normalization=[]
g_feature=[]
path=r'E:\BaiduSyncdisk\machine learning test\task1\样例'
samples = os.listdir(path)  # 函数返回值是一个列表，其中各元素均为字符串，分别是各路径名和文件名。
#samples.pop(0)#针对隐藏文件
for i in range(len(samples)):
    line_data=[]
    # data_dir = path+'/'+data_labels[i]    #遍历每个文件夹(假设数据存在默认路径的子文件夹)
    # datas = os.listdir(data_dir) #得到每个文件夹里的文件名称 存在datas列表中
    sample_path=path+'/'+samples[i]
    f = open(sample_path, encoding='ansi')
    lines = f.readlines()
    for line in lines:
        if line.isspace() or line.startswith('#'):  # 如果读到空行或者#号开头，就跳过
            continue
        else:
            line = line.replace("\n", "")  # 去除文本中的换行等等，可以追加其他操作
            line = line.replace("\t", "")
            line_data.append(line)  # 处理完成后的行，追加到列表中
    cur_sample_num=int(line_data[0])
    sample_num.append(cur_sample_num)   #保存样例个数
    line_data.pop(0)                  #删除样例个数行
    cur_feature_num=int(line_data[0]) #保存每个特征个数
    feature_num.append(cur_feature_num)
    line_data.pop(0)                 #删除特征个数行
    is_normalization.append(line_data[0]) #保存是否归一化行
    line_data.pop(0)                      #删除是否归一化行
    cur_g_feature=line_data[0] + ' ' + line_data[1]
    g_feature.append(cur_g_feature) #保存全局特征行
    line_data.pop(0)                        #删除全局特征行
    line_data.pop(0)
    assert cur_sample_num==len(line_data)  #判断样例个数行数值是否和样例个数相同
    sample_line_num_l=[]                   #获取最大样例个数
    for new_line_n in line_data:
        sample_line_num_l.append(int(new_line_n.split()[1]))
    for new_line in line_data:
        label.append(int(new_line.split()[0]))  #获取标签值
        sample_line_num = int(new_line.split()[1])
        new_line=' '.join(np.delete(new_line.split(),[0,1]).tolist())
        assert sample_line_num * cur_feature_num == len(new_line.split()) #判断每行样例个数是否和给定个数相同
        if sample_line_num < max(sample_line_num_l):
            new_line=new_line+' 0'*(max(sample_line_num_l)-sample_line_num)*cur_feature_num #补齐特征个数
        new_line=new_line+' '+cur_g_feature #添加全局特征
        new_line= list(map(float, new_line.split()))  #转化浮点数
        data.append(np.array(new_line))




