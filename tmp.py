import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 导入数据集
boston_house = tf.keras.datasets.boston_housing
(train_x, train_y), (test_x, test_y) = boston_house.load_data()
train_x = train_x[:, 12]
test_x = test_x[:, 12]
num_train = len(train_x)
num_test = len(test_x)

# 类型转换
X_train = tf.cast(train_x, tf.float32)
X_test = tf.cast(test_x, tf.float32)

Y_train = tf.cast(train_y, tf.float32)
Y_test = tf.cast(test_y, tf.float32)

# 设置超参数
learn_rate = 0.005
learn_count = 8000
ds = 100

# 设置模型变量初值
w = tf.Variable(np.random.randn(), dtype=tf.float32)
b = tf.Variable(np.random.randn(), dtype=tf.float32)

# 训练模型
mse_train = []
mse_test = []
for i in range(0, learn_count + 1):
    with tf.GradientTape() as tape:
        # PRED_train = tf.matmul(X_train,W)
        PRED_train = w * X_train + b
        Loss_train = 0.5 * tf.reduce_mean(tf.square(Y_train - PRED_train))

        PRED_test = w * X_test + b  # tf.matmul(X_test,W)
        Loss_test = 0.5 * tf.reduce_mean(tf.square(Y_test - PRED_test))

    mse_train.append(Loss_train)
    mse_test.append(Loss_test)

    dL_dw, dL_db = tape.gradient(Loss_train, [w, b])
    # print(dL_dw,dL_db)
    w.assign_sub(learn_rate * dL_dw)
    b.assign_sub(learn_rate * dL_db)

    if i % ds == 0:
        print("i:", i, "训练误差: ", Loss_train.numpy(), "测试误差:", Loss_test.numpy())
print("w:", w.numpy(), "\nb", b.numpy())

x_test = np.array(test_x.reshape(-1))
y_pred = (w * x_test + b).numpy()

plt.figure()
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.scatter(train_x, train_y, color="red", label="销售记录")
plt.scatter(x_test, y_pred, color="blue", label="预测房价")
plt.plot(x_test, y_pred, color="green", label="拟合直线", linewidth=2)

plt.xlabel("LSTAT(%)", fontsize=14)
plt.ylabel("价格(万元)", fontsize=14)

plt.xlim((0, 50))
plt.ylim((0, 50))

plt.legend(loc="upper left")
plt.show()
