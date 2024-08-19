import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import time
import sys

# from .. logger import get_logger
# from .. utils import min_max_normalization

# 未处理除零情况
# def min_max_normalization(x: np.ndarray):
#     if not isinstance(x, np.ndarray):
#         print("min_max_normalization: input must be np.ndarray")
#     if x.ndim == 1:
#         return (x-x.min())/(x.max()-x.min())
#     else:
#         x = x.T
#         x = (x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0))
#         return x.T

def min_max_normalization(x: np.ndarray):
    if not isinstance(x, np.ndarray):
        raise TypeError("min_max_normalization: input must be np.ndarray")

    if x.ndim == 1:
        min_val = np.nanmin(x)
        max_val = np.nanmax(x)
        range_val = max_val - min_val
        if range_val == 0:
            range_val = 1e-10
        return (x - min_val) / range_val
    else:
        min_val = np.nanmin(x, axis=0)
        max_val = np.nanmax(x, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1e-10
        return (x - min_val) / range_val

BATCH_SIZE = 32
INIT_LR = 1e-4

# logger = get_logger("classification")

class DataSetCNN(Dataset):
    def __init__(self, data, label, length=30, eval=False):
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if not eval:
            assert(data.shape[0] == label.shape[0])
            self.datas = data
            self.label = label
        else:
            self.datas = data
        self.length = length
        self.eval = eval

    def __len__(self):
        return self.datas.shape[0]

    def __getitem__(self, i):
        if self.eval:
            return torch.tensor(self.datas[i].reshape(-1, self.length), dtype=torch.float)
        else:
            return torch.tensor(self.datas[i].reshape(-1, self.length), dtype=torch.float), torch.tensor(self.label[i], dtype=torch.long)


class CNN(nn.Module):
    def __init__(self, classes=50):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, 5, 1, 1),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # nn.AdaptiveAvgPool1d(1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 5, 1, 1, bias=False),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, 5, 1, 1, bias=False),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.flatten = nn.Sequential(
            nn.Linear(512, 64), nn.BatchNorm1d(64), nn.ReLU())
        # self.out = nn.Linear(256, 64)
        self.out = nn.Linear(64, classes)
        self.dropout = nn.Dropout(0.5)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # print(x)
        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.flatten(x)
        # x=torch.cat((x,y),1)
        # x=self.avgpool(x)
        x = self.out(x)
        # x=self.out1(x)
        x = nn.LogSoftmax(dim=-1)(x)
        # x=nn.Softmax(dim=-1)(x)
        return x


class CNNClassifier(object):
    def __init__(self, model=None, class_num=13, length=30):
        """
        Parameters:
            model: 如果是训练好的模型，这里直接赋值，否则不传参
            class_num: 类别
        """
        if model is None:
            self.model = CNN(class_num)
        else:
            self.model = model
        self.class_num = class_num
        self.length = length

    def fit(self, X: np.ndarray, y: np.array, max_epoch=100, stop_epoch=10):
        """
        训练模型
        Parameters:
            X: 数据
            y: label
            max_epoch: 最多训练多少epoch
            stop_epoch: 多少个epoch在valid data上无提升，则停止训练
        """
        print("start training")
        # ------init--------
        initial_lr = INIT_LR
        batch_size = BATCH_SIZE
        criterion = nn.CrossEntropyLoss()
        count = 0
        train_X, valid_X, train_y, valid_y = train_test_split(
            X, y, test_size=0.3)
        train_data, valid_data = DataSetCNN(train_X, train_y, length=self.length), DataSetCNN(
            valid_X, valid_y, length=self.length)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=initial_lr)
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(
            valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
        is_gpu = torch.cuda.is_available()
        if is_gpu:
            self.model = self.model.cuda()
            criterion = criterion.cuda()
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.8)
        max_f, best_p, best_r = 0, 0, 0
        best_model = None

        # ------train------
        for epoch in range(max_epoch):
            y_true, y_pred = np.array(
                [], dtype=np.int), np.array([], dtype=np.int)
            self.model.train()
            # for b in tqdm(train_loader):
            for _, b in enumerate(train_loader):
                optimizer.zero_grad()
                if is_gpu:
                    b[0] = b[0].cuda()
                    b[1] = b[1].cuda()
                out = self.model(b[0])
                batch_pred = out.data.max(1)[1]
                loss = criterion(out, b[1])
                loss.backward()
                optimizer.step()
                if is_gpu:
                    b[1] = b[1].cpu()
                    batch_pred = batch_pred.cpu()
                y_true = np.append(y_true, b[1].numpy())
                y_pred = np.append(y_pred, batch_pred.numpy())
            p1, r1, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=np.unique(y_pred), average="macro")
            scheduler.step()

        # ----------validation-------------
            with torch.no_grad():
                self.model.eval()
                y_true, y_pred = np.array(
                    [], dtype=np.int), np.array([], dtype=np.int)
                for _, b in enumerate(valid_loader):
                    if is_gpu:
                        b[0] = b[0].cuda()
                        b[1] = b[1].cuda()
                    out = self.model(b[0])
                    batch_pred = out.data.max(1)[1]
                    if is_gpu:
                        b[1] = b[1].cpu()
                        batch_pred = batch_pred.cpu()
                    y_true = np.append(y_true, b[1].numpy())
                    y_pred = np.append(y_pred, batch_pred.numpy())
                p2, r2, f2, _ = precision_recall_fscore_support(
                    y_true, y_pred, labels=np.unique(y_pred), average="macro")
                if f2 > max_f:
                    best_p = p2
                    best_r = r2
                    max_f = f2
                    best_model = self.model
                    count = 0
                else:
                    count += 1
                    if count >= stop_epoch:
                        break
            print("Epoch:{}, Loss:{:.5f}\ntrain_p:{:.5f}, train_r:{:.5f}, train_f:{:.5f},valid_p:{:.5f}, valid_r:{:.5f}, valid_f:{:.5f},best_p:{:.5f}, best_r:{:.5f}, best_f:{:.5f}".format(
               epoch+1, loss.item(), p1, r1, f1, p2, r2, f2, best_p, best_r, max_f))
        self.model = best_model

    def predict(self, X: np.ndarray) -> np.array:
        """
        Parameters:
            X:待预测的数据
        Returns:
            pred:预测结果
        """
        if self.model is None:
            raise ValueError("please load or fit model first")

        test_data = DataSetCNN(min_max_normalization(
            X), None, eval=True, length=self.length)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
        self.model.eval()
        pred = np.array([], dtype=np.int)
        for _, b in enumerate(test_loader):
            out = self.model(b)
            batch_pred = out.data.max(1)[1]
            pred = np.append(pred, batch_pred.numpy())
        # if pred.shape[0] == 1:
        #     pred = pred[0]
        return pred

    def save_model(self, path):
        """
        Parameters:
            path:模型存储地址，强制加.pt后缀
        """
        if ".pt" not in path:
            path += ".pt"
        torch.save(self.model.state_dict(), path)
        # logger.info("save model successfully")

    def load_model(self, path):
        """
        Parameters:
            path:模型存储地址，强制加.pt后缀
        """
        if ".pt" not in path:
            path += ".pt"
        self.model.load_state_dict(torch.load(path))
        # logger.info("load model successfully")


if __name__ == "__main__":
    print("success")
