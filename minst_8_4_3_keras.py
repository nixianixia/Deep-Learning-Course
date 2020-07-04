'''
    tensorflow2.0
    minst 手写字符集
    使用 keras 实现两个隐藏层
    采用Keras序列模型进行建模与训练过程一般分为六个步骤：
    （1）创建一个Sequential模型；
    （2）根据需要，通过“add()”方法在模型中添加所需要的神经网络层，完成模型构建；
    （3）编译模型，通过“compile()”定义模型的训练模式；
    （4）训练模型，通过“fit()”方法进行训练模型；
    （5）评估模型，通过“evaluate()”进行模型评估；
    （6）应用模型，通过“predict()”进行模型预测。
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels_ohe = tf.one_hot(train_labels, depth=10).numpy()
test_labels_ohe = tf.one_hot(test_labels, depth=10).numpy()

#  创建一个Sequential模型；
model = tf.keras.models.Sequential()

# 添加输入（平坦）层，平坦层，将除了最高维的其它维度，变成一维，结果 总共是二维的
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # 样本数量是最高维

# 添加密集层，稠密层，全连接层 Dense 层
model.add(
    tf.keras.layers.Dense(
        units=64,  # 神经元数量
        kernel_initializer='normal',  # 权重初始化方式
        activation='relu'  # 激活函数
    ))

# 添加全连接层 2
model.add(
    tf.keras.layers.Dense(
        units=32,  # 神经元数量
        kernel_initializer='normal',  # 权重初始化方式
        activation='relu'  # 激活函数
    ))

# 添加输出层，但这也是一个全连接层
model.add(
    tf.keras.layers.Dense(
        units=10,  # 神经元数量
        activation='softmax'  # 激活函数
    ))

# 输出模型摘要
model.summary()

# 上面的模型也可以一次性建立
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(64, activation=tf.nn.relu),
#     tf.keras.layers.Dense(32, activation=tf.nn.relu),
#     tf.keras.layers.Dense(10, activation=tf.nn.softmax),
# ])

# 定义训练模式，这几个可参数可以接收名字，也可以接收实例
model.compile(
    optimizer='adam',  # 优化器, tf.keras.optimizers
    loss='categorical_crossentropy',  # 损失函数  tf.keras.losses
    metrics=['accuracy']  # 评估模型的方式， loss 是默认添加的    tf.keras.metrics
)

# 设置训练参数
train_epochs = 10  # 训练轮次
batch_size = 30  # 批次大小

# 模型训练
train_history = model.fit(
    train_images,
    train_labels_ohe,
    validation_split=0.2,  # 验证集划分
    epochs=train_epochs,  # 训练轮次
    batch_size=batch_size,  # 批次大小
    verbose=2  # 日志显示方式
)

# tf.keras.Model.fit()常见参数：
# x ：训练数据；
# y ：目标数据（数据标签）；
# epochs ：将训练数据迭代多少遍；
# batch_size ：批次的大小；
# validation_data ：验证数据，可用于在训练过程中监控模型的性能。
# verbose：训练过程的日志信息显示，0为不在标准输出流输出日志信息，1为输
# 出进度条记录，2为每个epoch输出一行记录。


def show_train_history(train_history, train_metric, val_metric):
    plt.plot(train_history.history[train_metric])
    plt.plot(train_history.history[val_metric])
    plt.title("Train History")
    plt.ylabel(train_metric)
    plt.xlabel("Epoch")
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels_ohe, verbose=2)

print(test_loss, test_acc)
print(model.metrics_names)

# 使用模型
# 进行预测
test_pred = model.predict(test_images)  # 结果以独热编码返回
print("predict : ", test_pred.shape, "  第10个图片的的预测值：", np.argmax(test_pred[9]),
      "标签值： ", test_labels[9])

# 进行分类预测
test_pred = model.predict_classes(test_images)  # 只接返回预测结果
print("predict : ", test_pred.shape, "  第10个图片的的预测值：", test_pred[9], "标签值： ",
      test_labels[9])

# show_train_history(train_history, 'loss', 'val_loss')
# show_train_history(train_history, 'accuracy', 'val_accuracy')

# 面向整数标签的序列模型构建与训练
# 针对采用整数类型的标签类别数据，Keras提供了更为简便的方法，无需针
# 对这些标签数据先进行独热编码就能直接应用
# 采用“sparse_categorical_crossentropy”损失函数来替换
# “categorical_crossentropy”损失函数
# loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
# loss = tf.keras.losses.categorical_crossentropy(
# y_true=tf.one_hot(y, depth=tf.shape(y_pred)[-1]),
# y_pred=y_pred
# )