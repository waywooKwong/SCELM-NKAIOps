import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

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

class DataSetCNN(Dataset):
    def __init__(self, data, label=None, length=20, eval=False):
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
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 5, 1, 1, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, 5, 1, 1, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.flatten = nn.Sequential(
            nn.Linear(512, 64), nn.BatchNorm1d(64), nn.ReLU())
        self.out = nn.Linear(64, classes)
        self.dropout = nn.Dropout(0.5)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.out(x)
        x = nn.LogSoftmax(dim=-1)(x)
        return x

class CNNClassifier(object):
    def __init__(self, model=None, class_num=13, length=20):
        if model is None:
            self.model = CNN(class_num)
        else:
            self.model = model
        self.class_num = class_num
        self.length = length

    def fit(self, X: np.ndarray, y: np.array, max_epoch=100, stop_epoch=10, incremental=False):
        print("start training")
        initial_lr = INIT_LR * 0.1 if incremental else INIT_LR
        batch_size = BATCH_SIZE
        criterion = nn.CrossEntropyLoss()
        count = 0
        train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.3)
        train_data, valid_data = DataSetCNN(train_X, train_y, length=self.length), DataSetCNN(valid_X, valid_y, length=self.length)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=initial_lr)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
        is_gpu = torch.cuda.is_available()
        if is_gpu:
            self.model = self.model.cuda()
            criterion = criterion.cuda()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
        max_f, best_p, best_r = 0, 0, 0
        best_model = None

        for epoch in range(max_epoch):
            y_true, y_pred = np.array([], dtype=np.int), np.array([], dtype=np.int)
            self.model.train()
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
            p1, r1, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=np.unique(y_pred), average="macro")
            scheduler.step()

            with torch.no_grad():
                self.model.eval()
                y_true, y_pred = np.array([], dtype=np.int), np.array([], dtype=np.int)
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
                p2, r2, f2, _ = precision_recall_fscore_support(y_true, y_pred, labels=np.unique(y_pred), average="macro")
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
        if self.model is None:
            raise ValueError("please load or fit model first")
        test_data = DataSetCNN(min_max_normalization(X), None, eval=True, length=self.length)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
        self.model.eval()
        pred = np.array([], dtype=np.int)
        for _, b in enumerate(test_loader):
            out = self.model(b)
            batch_pred = out.data.max(1)[1]
            pred = np.append(pred, batch_pred.numpy())
        return pred

    """
    classifier = CNNClassifier(class_num=13, length=20)  # 初始化分类器

    pretrained_model_path = "path/to/your/pretrained_model.pt"  # 加载预训练模型
    classifier.load_model(pretrained_model_path)

    classifier.fit(X_new, y_new, max_epoch=50, stop_epoch=10, incremental=True)  # 进行初始训练

    classifier.fine_tune_with_misclassifications(X_new, y_new, epochs=5)  # 使用误分类数据进行微调

    incremental_model_path = "path/to/your/incremental_model.pt"  # 保存增量训练后的模型
    classifier.save_model(incremental_model_path)

    """

    def fine_tune_with_misclassifications(self, X: np.ndarray, y: np.array, epochs=10):
        pred = self.predict(X)
        misclassified_idx = np.where(pred != y)[0]
        if len(misclassified_idx) == 0:
            print("No misclassifications to fine-tune on.")
            return

        misclassified_X = X[misclassified_idx]
        misclassified_y = y[misclassified_idx]

        misclassified_dataset = DataSetCNN(misclassified_X, misclassified_y, length=self.length)
        misclassified_loader = DataLoader(misclassified_dataset, batch_size=BATCH_SIZE, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=INIT_LR * 0.1)
        is_gpu = torch.cuda.is_available()
        if is_gpu:
            self.model = self.model.cuda()
            criterion = criterion.cuda()

        self.model.train()
        for epoch in range(epochs):
            for _, b in enumerate(misclassified_loader):
                optimizer.zero_grad()
                if is_gpu:
                    b[0] = b[0].cuda()
                    b[1] = b[1].cuda()
                out = self.model(b[0])
                loss = criterion(out, b[1])
                loss.backward()
                optimizer.step()
            print(f"Fine-tuning epoch {epoch+1}/{epochs} completed.")

    def save_model(self, path):
        if ".pt" not in path:
            path += ".pt"
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        if ".pt" not in path:
            path += ".pt"
        # 只加载预训练模型中与当前模型匹配的部分权重，并忽略不匹配的部分。你可以使用 strict=False 参数来部分加载权重。
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict, strict=False)

if __name__ == "__main__":
    print("success")


########################### test #########################
import pandas as pd

def load_data_from_csv(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 提取第3到第32列作为特征X_new
    X_new = df.iloc[:, 3:23].values
    
    # 提取第34列（名为 "true-label"）作为标签y_new
    y_new = df['true-label'].values
    
    return X_new, y_new

# 假设已经定义并导入了CNNClassifier类
classifier = CNNClassifier(class_num=15, length=20)  # 初始化分类器

pretrained_model_path = '/home/sunyongqian/liuheng/Time-LLM/weihua/AreaText/data/k5_finetune_modify/weihua_train2.pt'  # 加载预训练模型
classifier.load_model(pretrained_model_path)

# 读取CSV文件并提取X_new和y_new
input_csv_path = '/home/sunyongqian/liuheng/Time-LLM/weihua/AreaText/data/k5_finetune_modify/modify_40001.csv'
X_new, y_new = load_data_from_csv(input_csv_path)

# 使用误分类数据进行微调
classifier.fine_tune_with_misclassifications(X_new, y_new, epochs=5)

# 保存增量训练后的模型
incremental_model_path = "/home/sunyongqian/liuheng/Time-LLM/weihua/AreaText/data/k5_finetune_modify/incremental_model3_20.pt"
classifier.save_model(incremental_model_path)
