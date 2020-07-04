import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x1 = tf.constant([137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00,
                106.69, 138.05, 53.75, 46.91, 68.00, 63.02, 81.26, 86.21])
x2 = tf.constant([3, 2, 2, 3, 1, 2, 3, 2, 2, 3, 1, 1, 1, 1, 2, 2],dtype = tf.float32)
y = tf.constant([145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 91.00,
                62.00, 133.00, 51.00, 45.00, 78.50, 69.65, 75.69, 95.30])

x0 = tf.ones(len((x1)),dtype = tf.float32)
X = tf.stack((x0,x1,x2),axis = 1)
Y = tf.reshape(y,(-1,1))

Xt = tf.transpose(X)
XtX_1 = tf.linalg.inv(tf.matmul(Xt,X))
XtX_1_Xt = tf.matmul(XtX_1,Xt)
W = tf.matmul(XtX_1_Xt , Y)

plt.rcParams['font.sans-serif'] = ['SimHei']

fig = plt.figure(figsize = (8,6))
ax3d = Axes3D(fig)
ax3d.scatter(x1,x2,y,color = "b",marker="*")

ax3d.set_xlabel('Area',color = 'r',fontsize = 16)
ax3d.set_ylabel('Room',color = 'r' ,fontsize = 16)
ax3d.set_zlabel('Price',color = 'r' ,fontsize = 16)
ax3d.set_yticks([1,2,3])
ax3d.set_zlim3d(30,160)

X1,X2 = tf.meshgrid(x1,x2)
Y_PRED = W[1] * X1 + W[2] * X2 + W[0]

fig2 = plt.figure()
ax3d2 = Axes3D(fig2)
ax3d2.plot_surface(X1,X2,Y_PRED,cmap='coolwarm')

ax3d2.set_xlabel('Area',color = 'r',fontsize = 14)
ax3d2.set_ylabel('Room',color = 'r', fontsize = 14)
ax3d2.set_zlabel('Price',color = 'r', fontsize = 14)
ax3d2.set_yticks([1,2,3])

y_pred = W[1] * x1 + W[2] * x2 + W[0]
fig3 = plt.figure()
ax3d3 = Axes3D(fig3)
ax3d3.scatter(x1,x2,y,color='b',marker='*',label='销售记录')
ax3d3.scatter(x1,x2,y_pred,color = 'r',label = '预测房价')
ax3d3.plot_wireframe(X1,X2,Y_PRED,color='c',linewidths=0.5,label="拟合平面")
ax3d3.set_xlabel('Area',color='r',fontsize= 14)
ax3d3.set_ylabel('Room',color = 'r',fontsize = 14)
ax3d3.set_zlabel('Price',color='r',fontsize=14)
ax3d3.set_yticks([1,2,3])

plt.suptitle('商品房价销售回归模型',fontsize = 20)
plt.legend(loc='upper left')

plt.show()