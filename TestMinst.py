'''
    这是单一神经网络的minst分类问题
    
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

#数据塑形
train_x = train_x.reshape(-1, 784)
valid_x = valid_x.reshape(-1, 784)
test_x = test_x.reshape(-1, 784)
print(train_x.shape, valid_x.shape, test_x.shape)

#数据归一化
train_x = tf.cast(train_x / 255.0, tf.float32)
valid_x = tf.cast(valid_x / 255.0, tf.float32)
test_x = tf.cast(test_x / 255.0, tf.float32)

#对标签进行独热编码
train_y = tf.one_hot(train_y, depth=10)
valid_y = tf.one_hot(valid_y, depth=10)
test_y = tf.one_hot(test_y, depth=10)


#定义模型
def model(x, w, b):
    pred = tf.matmul(x, w) + b
    return tf.nn.softmax(pred)


#定义均方差交叉熵损失函数
def loss(x, y, w, b):
    pred = model(x, w, b)
    loss_ = tf.keras.losses.categorical_crossentropy(
        y_true=y, y_pred=pred)  # y * tf.math.log(pred)
    return tf.reduce_mean(loss_)


#定义梯度计算函数
def grad(x, y, w, b):
    with tf.GradientTape() as tape:
        loss_ = loss(x, y, w, b)
    return tape.gradient(loss_, [w, b])  #返回梯度向量


# 定义准确率
def accuracy(x, y, w, b):
    pred = model(x, w, b)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 定义超参数
training_epochs = 20  # 训练轮数
batch_size = 50  #单次训练样本数
learn_rate = 0.001

# 初始模型参数
W = tf.Variable(
    tf.random.normal([784, 10], mean=0.0, stddev=1.0, dtype=tf.float32))
B = tf.Variable(tf.zeros([10], dtype=tf.float32))

# 定义优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)

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
        optimizer.apply_gradients(zip(grads, [W, B]))  #优化器根据梯度自动调整变量 w 和 b

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
