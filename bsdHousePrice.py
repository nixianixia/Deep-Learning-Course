import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#导入数据集
boston_house = tf.keras.datasets.boston_housing
(train_x,train_y),(test_x,test_y) = boston_house.load_data()
num_train = len(train_x)
num_test = len(test_x)

#数据归一化
x_train = (train_x - train_x.min(axis = 0))/(train_x.max(axis = 0) - train_x.min(axis = 0))
y_train = train_y

x_test = (test_x - test_x.min(axis = 0))/(test_x.max(axis = 0) - test_x.min(axis = 0))
y_test = test_y


x0_train = np.ones(num_train).reshape(-1,1)
x0_test = np.ones(num_test).reshape(-1,1)

#类型转换
X_train = tf.cast(tf.concat([x0_train, x_train],axis = 1),tf.float32)
X_test = tf.cast(tf.concat([x0_test, x_test],axis = 1), tf.float32)

Y_train = tf.constant(y_train.reshape(-1,1),tf.float32)
Y_test = tf.constant(y_test.reshape(-1,1),tf.float32)

#设置超参数
learn_rate = 0.01
learn_count = 2300
ds = 100

#设置模型变量初值
W = tf.Variable(np.random.randn(14,1),dtype=tf.float32)

#训练模型
mse_train =[]
mse_test = []
for i in range(0,learn_count+1):
	with tf.GradientTape() as tape:
		PRED_train = tf.matmul(X_train,W)
		Loss_train = 0.5 * tf.reduce_mean(tf.square(Y_train - PRED_train))
	
		PRED_test = tf.matmul(X_test,W)
		Loss_test = 0.5 * tf.reduce_mean(tf.square(Y_test - PRED_test))
	
	mse_train.append(Loss_train)
	mse_test.append(Loss_test)

	dL_dW = tape.gradient(Loss_train,W)
	print(type(dL_dW))
	W.assign_sub(learn_rate*dL_dW)

	if i % ds == 0:
		print("i:",i,"训练误差: ",Loss_train,"测试误差:",Loss_test)
print("W : ",W)