# 西安科技大学 11.2 节中一元逻辑回归实例
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# 特征：房间面积
x = np.array([
    137.97, 104.50, 100.00, 126.32, 79.20, 99.00, 124.00, 114.00, 106.69,
    140.05, 53.75, 46.91, 68.00, 63.02, 81.26, 86.21
])
# 标签：是否属于高档住宅
y = np.array([1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0])

plt.figure(figsize=(10, 5))
plt.rcParams['font.sans-serif'] = "SimHei"
plt.rcParams['axes.unicode_minus'] = False
plt.subplot(131)
plt.scatter(x, y)
plt.title("原始数据")

x_train = x - np.mean(x)  #把数据中心化，特征之间的间隔等比缩小
y_train = y
plt.subplot(132)
plt.scatter(x_train, y_train)
plt.title("中心化后的数据")

#设置超参数
learn_rate = 0.005
learn_num = 10
display_step = 1

#生成初始权重与偏置
np.random.seed(612)
w = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())

#绘制初始权值时 sigmoid 函数
x_ = range(-80, 80)
y_ = 1 / (1 + tf.exp(-(w * x_ + b)))
plt.subplot(133)
plt.plot(x_, y_, color="red", linewidth=3)
#训练模型

cross_train = []  #用来存放交叉熵损失
acc_train = []  #存放准确率

for i in range(learn_num + 1):
    with tf.GradientTape() as tape:
        #使用sigmoid函数计算预测概率
        pred_train = 1 / (1 + tf.exp(-(w * x_train + b)))
        #计算平均交叉熵损失
        Loss_train = -tf.reduce_mean(y_train * tf.math.log(pred_train) +
                                     (1 - y_train) *
                                     tf.math.log(1 - pred_train))
        #计算准确率
        Accuracy_train = tf.reduce_mean(
            tf.cast(tf.equal(tf.where(pred_train < 0.5, 0, 1), y_train),
                    tf.float32))
    #记录损失与准确率
    cross_train.append(Loss_train)
    acc_train.append(Accuracy_train)

    #梯度下降
    #w,b 与交叉熵损失函数求偏导数
    dL_dw, dL_db = tape.gradient(Loss_train, [w, b])
    #梯度方向调整模型参数
    w.assign_sub(learn_rate * dL_dw)
    b.assign_sub(learn_rate * dL_db)

    #显示训练信息
    if i % display_step == 0:
        print("i: %i,训练损失：%f, 准确率: %f" % (i, Loss_train, Accuracy_train))
        y_ = 1 / (1 + tf.exp(-(w * x_ + b)))
        plt.plot(x_, y_)
plt.show()
