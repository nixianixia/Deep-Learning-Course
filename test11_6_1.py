# 西大11.6节 实验
import tensorflow as tf
import numpy as np

# 独热编码示例
a = [0, 2, 3, 5]
b = tf.one_hot(a, 6)  # 参数1：一维张量，编码深度
print('a的狂热编码：', b)
print('--------------------分隔线--------------')

# 准确率示例
pred = np.array([[0.1, 0.2, 0.7], [0.1, 0.7, 0.2], [0.3, 0.4, 0.3]])  # 预测值
y = np.array([2, 1, 0])  # 标记
y_onehot = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])  # 独热编码

print("预测值中的最大数索引 ：", tf.argmax(pred, axis=1))
print("对比预测值与标记： ", tf.equal(tf.argmax(pred, axis=1), y))
print("转换对比结果：", tf.cast(tf.equal(tf.argmax(pred, axis=1), y), tf.float32))
print(
    "计算准确率：",
    tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), y), tf.float32)))

# 交叉熵损失函数
# print(tf.math.log(pred))
print(-y_onehot * tf.math.log(pred))
print("所有样本交叉熵之和：", -tf.reduce_sum(y_onehot * tf.math.log(pred)))
print("平均交叉熵损失：", -tf.reduce_sum(y_onehot * tf.math.log(pred)) / len(pred))
