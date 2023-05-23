import os
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
RANDOM_SEED = 625 # 固定随机种子
from sklearn import preprocessing #归一化

def load_data(path):
    data = []
    label = []
    g_feature = []
    samples = os.listdir(path)  # 函数返回值是一个列表，其中各元素均为字符串，分别是各路径名和文件名。
    for i in range(len(samples)):
        line_data = []
        normalization=[]
        # data_dir = path+'/'+data_labels[i]    #遍历每个文件夹(假设数据存在默认路径的子文件夹)
        # datas = os.listdir(data_dir) #得到每个文件夹里的文件名称 存在datas列表中
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
        cur_feature_num = int(line_data[1])  # 保存每个特征个数
        is_normalization=int(line_data[2])  # 保存是否归一化行
        cur_g_feature = line_data[3] + ' ' + line_data[4]
        g_feature.append(cur_g_feature)  # 保存全局特征行
        for j in range(5):
            line_data.pop(0)  # 删除样例个数行
        assert cur_sample_num == len(line_data)  # 判断样例个数行数值是否和样例个数相同
        sample_line_num_l = []  # 获取最大样例个数
        for new_line_n in line_data:
            sample_line_num_l.append(int((len(new_line_n.split())-1)/4))

        for new_line in line_data:
            label.append(int(new_line.split()[0]))  # 获取标签值
            sample_line_num = int((len(new_line.split())-1)/4)
            new_line = ' '.join(np.delete(new_line.split(), [0]).tolist())
            #assert sample_line_num * cur_feature_num == len(new_line.split())  # 判断每行样例个数是否和给定个数相同

            if sample_line_num < max(sample_line_num_l):
                new_line = new_line + ' 0' * (max(sample_line_num_l) - sample_line_num) * cur_feature_num  # 补齐特征个数
            new_line = new_line + ' ' + cur_g_feature  # 添加全局特征
            new_line = list(map(float, new_line.split()))  # 转化浮点数
            if is_normalization:
                data.append(np.array(new_line))
            else:
                normalization.append(np.array(new_line))
          #判断是否归一化，若已经归一化直接返回，若没有，则进行归一化操作
        if not is_normalization:
            min_max_scaler = preprocessing.MinMaxScaler()
            normalization= min_max_scaler.fit_transform(normalization)
            for ijk in normalization:
                data.append(np.array(ijk))
    return np.array(data),np.array(label)

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

path = r'D:\BaiduSyncdisk\machine learning test\task1\样例'
all_x,all_y=load_data(path)
# print(all_y)
# print(all_x)



x_train,x_test,y_train,y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=RANDOM_SEED)

# ks = [1, 3, 5, 7]   #测试模型
# score_l1, score_l1_w = knn_sklearn(x_train, y_train, ks, 1)
# score_l2, score_l2_w = knn_sklearn(x_train, y_train, ks, 2)
# print(score_l1, score_l1_w, score_l2, score_l2_w)


clf = KNeighborsClassifier(n_neighbors=3, weights='distance', p=1)
clf.fit(x_train, y_train)
res = clf.score(x_test, y_test)
print(" Accurancy:", res)


predict=[]
predict=clf.predict(all_x)

ori=all_y.tolist()
print(len(ori))
error=[]
for ind in range(len(ori)):
    if (ori-predict)[ind] !=0 :
        error.append(ind+1)
print("error sample index is:",error)
