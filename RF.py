import numpy as np
import os
import pickle
from joblib import dump, load
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score
from joblib import dump
import warnings
import numpy as np
warnings.filterwarnings('ignore')

file = open("g:/opt/opt/allfeatures.pkl", "rb")
AllFeatures = pickle.load(file)

n = 100
def merge(malvector, benvector):  # 数据操作，返回操作后的恶意样本
    AddFeatures = []
    size = len(malvector)
    newmalvector = []
    newbenvector = []
    for i in range(0, size):
        if i >= 4441:  # 设置特征界限
            newmalvector.append(malvector[i])
        else:
            if malvector[i] == 1:
                newmalvector.append(1)
            else:
                if benvector[i] == 0:
                    newmalvector.append(0)
                else:
                    newmalvector.append(1)
                    AddFeatures.append(AllFeatures[i])

    return newmalvector, AddFeatures

clf = load("../rf.pkl")
def predict(v):
    vec = []      #改变为二维向量
    vec.append(v)
    res = clf.predict(vec)
    return res[0]


def getScore(v1, v2):  # pre post 获取恶意性指标的度值
    # print equal(v1,v2)
    premal = []
    premal.append(v1)
    postmal = []
    postmal.append(v2)  # 改变为二维向量格式
    predistance = clf.predict_proba(premal)[0][1] * n
    postdistance =clf.predict_proba(postmal)[0][1] * n
    # predistance = clf.predict_log_proba(premal)[0][1]
    # postdistance =clf.predict_log_proba(postmal)[0][1]
    distance = predistance - postdistance
    return distance


# afterben = dir2vector("/home/huds/opt/benign")

def equal(v1, v2):  # 判断两向量是否相同
    flag = True
    for i in range(len(v1)):
        if v1[i] != v2[i]:
            flag = False
            break
    return flag


def crossover(v1, v2):  # 杂交 返回杂交后的向量
    res = []
    size = len(v1)
    for i in range(size):
        if v1[i] == 1 or v2[i] == 1:
            res.append(1)
        else:
            res.append(0)
    return res


def getFittest(benpopulation):  # 50 ge
    for i in range(0, 5):
        for j in range(i + 1, 5):
            benpopulation.append(crossover(benpopulation[i], benpopulation[j]))
    return benpopulation


newPool = [] #最后逃逸的向量


def Select(malpopulation, benpopulation):
    malsize = len(malpopulation)
    bensize = len(benpopulation)
    newmalpopulation = []
    dict = {}
    sum = [0 for i in range(0, len(benpopulation))]
    for i in range(0, malsize):
        #AddFeatures = [] 待添加的特征
        isSuccess = False
        for j in range(0, bensize): #计算良性化度值总和
            print(i,j)
            ans = merge(malpopulation[i], benpopulation[j])
            distance = getScore(malpopulation[i], ans[0])  # 设置
            sum[j] = sum[j] + distance
            res = predict(ans[0])
            if res == 0 and isSuccess == False:
                isSuccess = True
                newPool.append(ans[0])
                #AddFeatures.extend(ans[1])
        if isSuccess == True:
            print("---------------------Success! --------------" + str(i))
            # print("添加向量：")
            # for u in AddFeatures:
            #     print(u)
        else:
            newmalpopulation.append(malpopulation[i]) #剩下的恶意样本向量到下一轮
    for i in range(len(sum)):
        dict[i] = sum[i] / len(malpopulation)
    skey = sorted(dict, key=dict.__getitem__, reverse=True)
    newbenpopulation = []

    threshold = 3
    print("适应值前三名：")
    for i in range(threshold):
        print(dict[skey[i]])
        newbenpopulation.append(benpopulation[skey[i]])
    newbenpopulation = getFittest(newbenpopulation)
    return newmalpopulation, newbenpopulation


# Initial malvectors
malware = load("../newmal.pkl")
benign = load("../benign.pkl")
malpopulations = malware
print(len(malpopulations))
# Initial benvectors

batch_size = 100
i = 1 #每次加入种群中良性向量的次数
benpopulations = benign[0:batch_size]
print(len(benpopulations))
count = 0
for epoch in range(0, 15):
    malpopulations, benpopulations = Select(malpopulations, benpopulations)
    benpopulations.extend(benign[i * batch_size: i * batch_size + batch_size])
    i = i + 1
    print("剩余恶意样本：" + str(len(malpopulations)))
    count = count + 1
    print("第{v}次迭代".format(v=count))
    if len(malpopulations) == 0:
        break

dump(newPool, "../rf_GA.pkl")
 #剩50个 267
# 适应值前三名：
# 41.81045884673145
# 20.066701043984875
# 4.188299046481752
# 剩余恶意样本：86
# 第1次迭代

# 适应值前三名：
# 34.87596899224806
# 34.87596899224806
# 34.87596899224806
# 剩余恶意样本：65
# 第2次迭代

# 适应值前三名：
# 34.684615384615384
# 30.95897435897436
# 30.95897435897436
# 剩余恶意样本：54
# 第3次迭代


# 适应值前三名：
# 36.04320987654321
# 36.04320987654321
# 36.04320987654321
# 剩余恶意样本：50
# 第4次迭代

# 适应值前三名：
# 35.36666666666667
# 35.36666666666667
# 35.36666666666667
# 剩余恶意样本：50
# 第5次迭代