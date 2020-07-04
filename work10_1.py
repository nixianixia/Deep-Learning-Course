import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']

(train_x,train_y),(test_x,test_y)  = tf.keras.datasets.boston_housing.load_data()
train_x = train_x[:,12:]
test_x = test_x[:,12:]

x = tf.constant(train_x.reshape(-1))
y = tf.constant(train_y.reshape(-1))

meanX = tf.reduce_mean(x)
meanY = tf.reduce_mean(y)

sumXY = tf.reduce_sum((x-meanX)*(y-meanY))
sumX = tf.reduce_sum((x-meanX)*(x-meanX))

w = sumXY/sumX
b = meanY-w*meanX

print("权重W=",w.numpy(),"\n偏置值b =",b.numpy())
print("线性模型：y=",w.numpy(),"*x+",b.numpy())

x_test = np.array(test_x.reshape(-1))
y_pred = (w*x_test+b).numpy()

plt.figure()

plt.scatter(x,y,color="red",label="销售记录")
plt.scatter(x_test,y_pred,color='blue',label="预测房价")
plt.plot(x_test,y_pred,color='green',label="拟合直线",linewidth=2)

plt.xlabel("LSTAT",fontsize = 14)
plt.ylabel("价格(万元)",fontsize = 14)

plt.xlim((0,50))
plt.ylim((0,50))

plt.legend(loc = "upper left")
plt.show()