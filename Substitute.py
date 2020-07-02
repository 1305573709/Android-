from joblib import load, dump
import torch
import torch.nn as nn
import numpy as np
import numpy
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(4441, 8192),  # 2^13
            nn.ReLU(),
            nn.Linear(8192, 4441),
            nn.ReLU(),
            nn.Linear(4441, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layer.forward(x)
dnnmodel = torch.load("E:/pytorch/DNNModel.pkl")
def predict(vec):
    vec = torch.tensor(vec).cuda().float()
    pred = dnnmodel(vec).detach().cpu().numpy()
    out = torch.tensor(np.where(pred > 0.5, 1, 0))  # 1为恶意，0为良性
    res = out.float()
    return res
malware = load("E:/pytorch/malware.pkl")
benign = load("E:/pytorch/benign.pkl")
#clf = load("E:/pytorch/rf.pkl")
test_mal = load("E:/pytorch/newmal.pkl")
test_ben = load("E:/pytorch/newben.pkl")
train_x = malware + benign
# label = np.array([1] * len(malware) + [0] * len(benign))
label = numpy.array(predict(train_x))
per = np.random.permutation(label.shape[0])
#训练集
Train_X = []
for i in per:
    Train_X.append(train_x[i])
Train_Y = label[per]
#测试集
test_x = test_mal + test_ben
test_y = [1] * len(test_mal) + [0] * len(test_ben)

# DNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# device = "cpu"
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(4441, 8192),  # 2^13
            nn.ReLU(),
            nn.Linear(8192, 4441),
            nn.ReLU(),
            nn.Linear(4441, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layer.forward(x)


model = DNN().to(device)
loss_fn = nn.BCELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
x = torch.Tensor(Train_X).to(device)
y = torch.Tensor(Train_Y).to(device)
batch_size = 30
totalSteps = 9976
for epoch in range(30):
    for i in range(totalSteps // batch_size):
        start = i * batch_size
        end = start + batch_size if start + batch_size <= 9976 else 9976
        train_x = x[start:end].to(device)
        train_y = y[start:end].to(device)
        y_pred = model(train_x)
        loss = loss_fn(y_pred, train_y)
        print(epoch, i, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model, "sub_dnn.pkl")
