import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"  # 鸢尾花训练数据集下载地址
train_path = tf.keras.utils.get_file(
    TRAIN_URL.split("/")[-1], TRAIN_URL
)  # 下载训练数据集文件并保存

TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
test_path = tf.keras.utils.get_file(TEST_URL.split("/")[-1], TEST_URL)  # 保存测试集

df_iris = pd.read_csv(train_path, header=0)  # 读取训练数据集
test_df_iris = pd.read_csv(test_path, header=0)  # 读取测试集

iris = np.array(df_iris)  # 转化为numpy数组
train_x = iris[:, 0:2]  # 取出四个属性中的其中两个,第一个是花萼长度 ，第二个是花萼宽度
train_y = iris[:, 4]  # 取出样本的标签

test_iris = np.array(test_df_iris)  # 转numpy
test_x = test_iris[:, 0:2]  # 取测试集中花萼的长宽
test_y = test_iris[:, 4]  # 取测试集中标签

# 取出山鸢尾 与 变色鸢尾的数据
x_train = train_x[train_y < 2]
y_train = train_y[train_y < 2]

x_test = test_x[test_y < 2]
y_test = test_y[test_y < 2]

# 属性中心化
x_train = x_train - np.mean(x_train, axis=0)  # 按列中心化
x_test = x_test - np.mean(x_test, axis=0)

x0_train = np.ones(len(x_train)).reshape(-1, 1)  # 生成偏置项
x0_test = np.ones(len(x_test)).reshape(-1, 1)

X = tf.cast(tf.concat([x0_train, x_train], axis=1), tf.float32)  # 组合偏置与各个特征
X_TEST = tf.cast(tf.concat([x0_test, x_test], axis=1), tf.float32)
Y = tf.cast(y_train.reshape(-1, 1), tf.float32)
Y_TEST = tf.cast(y_test.reshape(-1, 1), tf.float32)

# 设置超参数
learn_rate = 0.2
learn_num = 120

display_step = 30

# 设置模型参数初始值
# np.random.seed(612)
W = tf.Variable(np.random.randn(3, 1), dtype=tf.float32)

plt.figure(figsize=(10, 4.5))
plt.subplot(132)
cm_pt = mpl.colors.ListedColormap(["blue", "red"])  # 自定义颜色集
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_pt)
x_ = [-1.5, 1.5]
y_ = -(W[1] * x_ + W[0]) / W[2]
plt.plot(x_, y_, color="r")
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])

# 训练模型
ce = []
TestCe = []
acc = []
TestAcc = []
for i in range(0, learn_num + 1):
    with tf.GradientTape() as tape:
        PRED = 1 / (1 + tf.exp(-tf.matmul(X, W)))
        PRED_TEST = 1 / (1 + tf.exp(-tf.matmul(X_TEST, W)))
        Loss = -tf.reduce_mean(Y * tf.math.log(PRED) + (1 - Y) * tf.math.log(1 - PRED))
        TestLoss = -tf.reduce_mean(
            Y_TEST * tf.math.log(PRED_TEST) + (1 - Y_TEST) * tf.math.log(1 - PRED_TEST)
        )

    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.where(PRED.numpy() < 0.5, 0.0, 1.0), Y), tf.float32)
    )
    TestAccracy = tf.reduce_mean(
        tf.cast(
            tf.equal(tf.where(PRED_TEST.numpy() < 0.5, 0.0, 1.0), Y_TEST), tf.float32
        )
    )
    ce.append(Loss)
    TestCe.append(Loss)
    acc.append(accuracy)
    TestAcc.append(TestAccracy)

    dL_dW = tape.gradient(Loss, W)
    W.assign_sub(learn_rate * dL_dW)

    if i % display_step == 0:
        print(
            "Train :  i: %i,Acc : %f , TestAccracy: %f,Loss:%f, TestLoss: %f"
            % (i, accuracy, TestAccracy, Loss, TestLoss)
        )
        # print("gradient: ",W)
        x_ = [-1.5, 1.5]
        y_ = -(W[1] * x_ + W[0]) / W[2]
        plt.plot(x_, y_)
plt.subplot(131)
plt.plot(ce, color="blue", label="Loss")
plt.plot(acc, color="red", label="Acc")

plt.subplot(133)
# 绘制分类图
M = 300

x1_min, x2_min = x_train.min(axis=0)
x1_max, x2_max = x_train.max(axis=0)
t1 = np.linspace(x1_min, x1_max, M)
t2 = np.linspace(x2_min, x2_max, M)
m1, m2 = np.meshgrid(t1, t2)

m0 = np.ones(M * M)

print(m0.shape,m1.shape,m2.shape)

X_mesh = tf.cast(
    np.stack([m0, m1.reshape(-1), m2.reshape(-1)], axis=1), dtype=tf.float32
)
Y_mesh = tf.cast(1 / (1 + tf.exp(-tf.matmul(X_mesh, W))), dtype=tf.float32)
Y_mesh = tf.where(Y_mesh < 0.5, 0, 1)

n = tf.reshape(Y_mesh, m1.shape)
cmpt = mpl.colors.ListedColormap(["blue", "red"])
cmbg = mpl.colors.ListedColormap(["#ffa0a0", "#a0ffa0"])

plt.pcolormesh(m1, m2, n, cmap=cmbg)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_pt)

plt.legend()
plt.show()
# print("iris.shape:",iris.shape)
# print("train_x.shape:",train_x.shape)
# print("train_y.shape:",train_y.shape)

# print('x_train.shape:',x_train.shape)
# print('y_train.shape:',y_train.shape)
