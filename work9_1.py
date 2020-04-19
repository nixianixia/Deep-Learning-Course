import tensorflow as tf

#面积
x1 = tf.constant([137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00,
                106.69, 138.05, 53.75, 46.91, 68.00, 63.02, 81.26, 86.21])
#房间数量
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

area = float(input("请输入面积"))
roomNum = int(input("请输入房间数量"))

if 20<=area<=500:
	if 1<=roomNum<=10:
		Y_PRED = W[1] * area + W[2] * roomNum + W[0]
		print("房间的售价大概在：",float(Y_PRED),"万左右")
	else:
		print("房间数量在1-10")
else:
	print("面积：20-500之间")