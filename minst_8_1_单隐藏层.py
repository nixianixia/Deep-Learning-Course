'''
    浙江城市学院
    全连接，单隐藏层神经网络
    minst 手写字符集 第个样本是一个 28 * 28 的灰度图，转为一维后，拥有 728 个输入特征
    tensorflow 2.0

    调试日志：
        1：在adam 优化器应用梯度时报错：
            解决：对 W , B 分别应用梯度
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 加载数据集（训练集与测试集）
minst = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = minst.load_data()

total_num = len(train_images)  # 样本总数
valid_split = 0.2  # 验证集百分比大小
train_num = int(total_num * (1 - valid_split))  # 训练集大小

# 划分训练集
train_x = train_images[:train_num]
train_y = train_labels[:train_num]

# 划分验证集
valid_x = train_images[train_num:]
valid_y = train_labels[train_num:]

# 定义别名测试集
test_x = test_images
test_y = test_labels

print(train_x.shape, valid_x.shape, test_x.shape)

# 数据塑形
train_x = train_x.reshape(-1, 784)
valid_x = valid_x.reshape(-1, 784)
test_x = test_x.reshape(-1, 784)
print(train_x.shape, valid_x.shape, test_x.shape)

# 数据归一化
train_x = tf.cast(train_x / 255.0, tf.float32)
valid_x = tf.cast(valid_x / 255.0, tf.float32)
test_x = tf.cast(test_x / 255.0, tf.float32)

# 对标签进行独热编码
train_y = tf.one_hot(train_y, depth=10)
valid_y = tf.one_hot(valid_y, depth=10)
test_y = tf.one_hot(test_y, depth=10)

# 定义第一隐藏层权重 和 偏置项变量
Input_Dim = 784                                 # 输入特征数量
H1_NN = 64                                      # 隐藏层的神经元数量
W1 = tf.Variable(tf.random.normal([Input_Dim, H1_NN], mean=0.0, stddev=1.0, dtype=tf.float32))
B1 = tf.Variable(tf.zeros([H1_NN], dtype=tf.float32))

# 定义输出层权重和偏置项变量
OutputDim = 10
W2 = tf.Variable(tf.random.normal([H1_NN, OutputDim], mean=0.0, stddev=1.0, dtype=tf.float32))
B2 = tf.Variable(tf.zeros([OutputDim], dtype=tf.float32))

# 建立待优化变量列表
W = [W1, W2]
B = [B1, B2]

# 定义模型前向计算
def model(x, w, b):
    x = tf.matmul(x, w[0]) + b[0]
    x = tf.nn.relu(x)
    x = tf.matmul(x, w[1]) + b[1]
    pred = tf.nn.softmax(x)
    return pred

# 定义交叉熵损失函数


def loss(x, y, w, b):
    pred = model(x, w, b)    # 计算模型预测值
    loss_ = tf.keras.losses.categorical_crossentropy(y_true=y, y_pred=pred)  # tensorflow 提供的交叉熵损失函数
    return tf.reduce_mean(loss_)    # 计算均方差

# 梯度计算函数


def grad(x, y, w, b):
    with tf.GradientTape() as tape:
        loss_ = loss(x, y, w, b)
    return tape.gradient(loss_, [w, b])     # 返回梯度向量

# 定义准确率


def accuracy(x, y, w, b):
    pred = model(x, w, b)       # 计算各个结果预测概率
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  # 检查匹配 预测概率 与 标签的匹配
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 将布尔类型转为浮点数，计算平均值


# 设置超参数


training_epochs = 20    # 训练轮数
batch_size = 50          # 批次大小
learning_rate = 0.01    # 学习率


# 定义优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# 训练模型
total_step = int(train_num / batch_size)  # 一轮训练多少次

loss_list_train = []  # 用于保存训练集loss值的列表
loss_list_valid = []  # 用于保存验证集loss值的列表
acc_list_train = []  # 用于保存训练集Acc值的列表
acc_list_valid = []  # 用于保存验证集的Acc的列表

for epoch in range(training_epochs):
    for step in range(total_step):
        xs = train_x[step * batch_size:(step + 1) * batch_size]
        ys = train_y[step * batch_size:(step + 1) * batch_size]

        grads = grad(xs, ys, W, B)  # 计算梯度
        # optimizer.apply_gradients(zip(grads, W+B))  # 优化器根据梯度自动调整变量 w 和 b
        optimizer.apply_gradients(zip(grads[0], W))
        optimizer.apply_gradients(zip(grads[1], B))
    loss_train = loss(train_x, train_y, W, B).numpy()  # 计算当前轮训练损失
    loss_valid = loss(valid_x, valid_y, W, B).numpy()  # 计算当前轮验证损失
    acc_train = accuracy(train_x, train_y, W, B).numpy()
    acc_valid = accuracy(valid_x, valid_y, W, B).numpy()

    loss_list_train.append(loss_train)
    loss_list_valid.append(loss_valid)
    acc_list_train.append(loss_train)
    acc_list_valid.append(loss_valid)
    print(
        "epoch = {:3d},train_loss={:.4f},train_acc={:.4f},val_loss={:.4f}.val_acc={:.4f}"
        .format(epoch + 1, loss_train, acc_train, loss_valid, acc_valid))