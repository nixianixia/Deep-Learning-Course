#西安科技大学 11.2 的实例，实现一元逻辑回归中的各种值的计算方法

import tensorflow as tf
import numpy as np

x = np.array([1,2,3,4],dtype=np.float32)#样本
y = np.array([0,0,1,1])#标签
w = tf.Variable(1.)#权重
b = tf.Variable(1.)#偏置
pred = 1/(1+tf.exp(-(w*x+b)))#sigmoid()函数，对数几率函数
print(pred)

eachLoss = y*tf.math.log(pred)+(1-y)*tf.math.log(1-pred)#计算的是各项损失
print("各项损失：",eachLoss)

allLoss = -tf.reduce_sum(eachLoss)#累计损失，交叉熵损失函数
print("累计损失:", allLoss)

meanLoss = -tf.reduce_mean(eachLoss) #平均交叉熵损失
print(meanLoss)

result = tf.where(pred < 0.5,0,1)#把概率转换为 0，1
print("类别结果：",result)

accuracy = tf.reduce_mean(tf.cast(result,tf.float32))
print("准确率",accuracy)

tf.where(pred>0.5,a,b)
	#a, b 还可以是数组，a，b 形状需要相同
	#缺省 a, b ,返回pred中 元素大于0.5的索引

