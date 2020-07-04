# 鸢尾花多分类实现
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"  # 鸢尾花训练数据集下载地址
train_path = tf.keras.utils.get_file(TRAIN_URL.split("/")[-1],
                                     TRAIN_URL)  # 下载训练数据集文件并保存

df_iris_train = pd.read_csv(train_path, header=0)
iris_train = np.array(df_iris_train)

x_train = iris_train[:, 2:4]  # 花瓣长度属性
y_train = iris_train[:, 4]  # 花瓣标签

num_train = len(x_train)  # 样本数

x0_train = np.ones(num_train).reshape(-1, 1)
X_train = tf.cast(tf.concat([x0_train, x_train], axis=1), tf.float32)  # 准备输入项
Y_train = tf.one_hot(tf.constant(y_train, dtype=tf.int32), 3)

learn_rate = 0.09
learn_num = 10000
display_step = 100

W = tf.Variable(np.random.randn(3, 3), dtype=tf.float32)

acc = []
cce = []

for i in range(0, learn_num + 1):
    with tf.GradientTape() as tape:
        PRED_train = tf.nn.softmax(tf.matmul(X_train, W))
        Loss_train = -tf.reduce_mean(
            Y_train * tf.math.log(PRED_train))
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(PRED_train.numpy(), axis=1), y_train),
                tf.float32)) 
    acc.append(accuracy)
    cce.append(Loss_train)
    dL_dw = tape.gradient(Loss_train, W)
    W.assign_sub(learn_rate * dL_dw)
    if i % display_step == 0:
        print("i:%i, Acc:%f, Loss:%f" % (i, accuracy, Loss_train))

M = 500
x1_min, x2_min = x_train.min(axis=0)
x1_max, x2_max = x_train.max(axis=0)

t1 = np.linspace(x1_min, x1_max, M)
t2 = np.linspace(x2_min, x2_max, M)

m1, m2 = np.meshgrid(t1, t2)

m0 = np.ones(M * M)
X_ = tf.cast(np.stack([m0, m1.reshape(-1), m2.reshape(-1)], axis=1),
             tf.float32)
Y_ = tf.nn.softmax(tf.matmul(X_, W))
Y_ = tf.argmax(Y_.numpy(), axis=1)

n = tf.reshape(Y_, m1.shape)

plt.figure(figsize=(8, 6))
cm_bg = mpl.colors.ListedColormap(['#A0ffa0', '#ffa0a0', '#a0a0ff'])

plt.pcolormesh(m1, m2, n, cmap=cm_bg)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='brg')

plt.show()