'''

'''

import tensorflow as tf
import matplotlib.pyplot as plt

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('train data shape: ', x_train.shape)
print('train label shape: ', y_train.shape)
print('test data shape: ', x_test.shape)
print('test label shape: ', y_test.shape)

label_dict = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

plt.imshow(x_train[6])
print(label_dict[y_train[6][0]])

# 对图像数据 归一化
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 建立Sequential纯属堆叠模型
model = tf.keras.models.Sequential()

# 卷积层 1
model.add(
    tf.keras.layers.Conv2D(filters=32,
                           kernel_size=(3, 3),
                           input_shape=(32, 32, 3),
                           activation='relu',
                           padding='same'))

# 防止过拟合
model.add(tf.keras.layers.Dropout(rate=0.3))

# 池化层 1
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# 卷积层2
model.add(
    tf.keras.layers.Conv2D(filters=64,
                           kernel_size=(3, 3),
                           activation='relu',
                           padding='same'))

# 防止过拟合
model.add(tf.keras.layers.Dropout(rate=0.3))

# 池化层 2
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# 平坦层
model.add(tf.keras.layers.Flatten())

# 全连接层
model.add(tf.keras.layers.Dense(units=128, activation='relu'))

# 输出层
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

# 设置训练参数
train_epochs = 50
batch_size = 100
# 模型设置
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 准备回调
logdir = './cifar10logs'  # 模型训练日志的保存路径
checkpoint_path = './checkpoint/Cifar10.{epoch:02d}-{val_loss:.2f}.H5'  # 检查点路径

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=2),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                       save_weights_only=True,
                                       verbose=0,
                                       save_freq='epoch'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
]

# 尝试加载模型
checkpoint_dir = './checkpoint/'

# 得到最新的检查点文件
model_filename = tf.train.latest_checkpoint(checkpoint_dir)
if model_filename != None:
    model.load_weights(model_filename)
    print("{}加载成功".format(model_filename))
else:
    print("未找到权重文件，需要重新开始训练")

# 训练模型
train_history = model.fit(x=x_train,
                          y=y_train,
                          validation_split=0.2,
                          epochs=train_epochs,
                          batch_size=batch_size,
                          callbacks=callbacks,
                          verbose=2)

print(model.summary())
