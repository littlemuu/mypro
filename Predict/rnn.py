import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import pymysql

from io import BytesIO
import base64

conn=pymysql.connect(
    host='127.0.0.1',
    user='root',
    password='123456',
    database='school',
)

query="SELECT test_time,avg_grade from predict_prediction"

df=pd.read_sql(query,con=conn)
# df = pd.read_csv("D://college3/111/data/Alcohol_Sales.csv", index_col = 0, parse_dates = True)
# df.dropna(inplace=True)\
conn.close()

df['test_time'] = pd.to_datetime(df['test_time'])
# 提取成绩值
y = df['avg_grade'].values.astype(float)

# 定义测试集大小为12
test_size = 12

# 选择y中除了最后12个数据外的其他数据作为训练集
train_set = y[:-test_size]
# 选择y中最后12个数据作为测试集
test_set = y[-test_size:]

from sklearn.preprocessing import MinMaxScaler

# 数据缩放
scaler = MinMaxScaler(feature_range=(-1, 1))

# 数据归一化
train_norm = scaler.fit_transform(train_set.reshape(-1, 1))

# 将归一化后的数据转换为tensor
train_norm = torch.FloatTensor(train_norm).view(-1)

# 定义窗口大小
window_size = 12
# 定义一个函数，用于生成输入数据和相应的标签数据，通过滑动窗口的方式遍历序列数据，并将每个窗口中的数据作为输入，
# 以及该窗口的下一个值作为相应的标签。这样生成的输入输出对被存储在一个列表中，并作为函数的返回值。
def input_data(seq,ws):
    out = []
    L = len(seq)
    for i in range(L-ws):
        window = seq[i:i+ws]
        label = seq[i+ws:i+ws+1]
        out.append((window,label))
    return out

train_data = input_data(train_norm, window_size)


class LSTMnetwork(nn.Module):
    # 定义初始化方法
    def __init__(self, input_size=1, hidden_size=100, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size

        # 定义LSTM层，将作为RNN模型的隐藏层，负责处理输入序列的时间依赖性，并将隐藏状态传递给下一个时间步
        self.lstm = nn.LSTM(input_size, hidden_size)

        # 将LSTM层的输出转换为模型的最终输出
        self.linear = nn.Linear(hidden_size, output_size)

        # 使用两个全零张量，一个用于细胞状态，另一个用于隐藏状态
        self.hidden = (torch.zeros(1, 1, self.hidden_size),
                       torch.zeros(1, 1, self.hidden_size))

    # 定义模型的前向传播过程
    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(
            seq.view(len(seq), 1, -1), self.hidden)
        pred = self.linear(lstm_out.view(len(seq), -1))
        return pred[-1]

# 设置随机种子，使实验结果能重现
torch.manual_seed(42)

model = LSTMnetwork()

criterion = nn.MSELoss()

# 创建一个Adam优化器对象，用于更新模型参数以最小化损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100

import time

start_time = time.time()

for epoch in range(epochs):
    for seq, y_train in train_data:
        optimizer.zero_grad()  # 梯度清零
        model.hidden = (torch.zeros(1, 1, model.hidden_size),  # 重置隐藏状态
                        torch.zeros(1, 1, model.hidden_size))

        y_pred = model(seq)  # 前向传播，得到预测值

        loss = criterion(y_pred, y_train)  # 计算损失函数
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数


future = 12

preds = train_norm[-window_size:].tolist()

model.eval()

for i in range(future):
    seq = torch.FloatTensor(preds[-window_size:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1,1,model.hidden_size),
                        torch.zeros(1,1,model.hidden_size))
        preds.append(model(seq).item())

true_predictions = scaler.inverse_transform(np.array(preds[window_size:]).reshape(-1, 1))
#print(true_predictions)

def draw1():
    X=np.arange('2014-12-31','2020-12-31',dtype='datetime64[M]')
    x = np.arange('2019-12-01', '2020-12-31', dtype='datetime64[M]').astype('datetime64[D]')
    plt.figure(figsize=(12,4))
    plt.title('Grades')
    plt.ylabel('avg_grade')
    plt.grid(True)
    #plt.autoscale(axis='x', tight=True)
    plt.plot(X,df['avg_grade'], color='#8000ff')
    plt.plot(x,true_predictions, color='#ff8000')
    #plt.show()
     # 将图像保存为字节流
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()  # 在编码之前关闭图形对象
    # 将字节流编码为base64字符串
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{image_base64}'

def draw2():
    x = np.arange('2019-12-01', '2020-12-31', dtype='datetime64[M]').astype('datetime64[D]')
    fig = plt.figure(figsize=(12,4))
    plt.title('Grades')
    plt.ylabel('avg_grade')
    plt.grid(True)
    plt.autoscale(axis='x',tight=True)
    fig.autofmt_xdate()
    df.index = pd.to_datetime(df.index)  # 将索引转换为日期时间类型
    plt.plot(df['avg_grade']['2020-01-01':], color='#8000ff')  # 使用日期时间索引进行切片操作
    plt.plot(x,true_predictions, color='#ff8000')
    #plt.show()
     # 将图像保存为字节流
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()  # 在编码之前关闭图形对象
    # 将字节流编码为base64字符串
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{image_base64}'

