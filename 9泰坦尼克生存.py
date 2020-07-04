'''
    泰坦尼克生存预测
    keras 完成
    预测 Jack 和 Rose
    模型训练可视化
    模型训练的回调与模型存储
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import urllib.request
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing

# 下载泰坦尼克数据集， 是一个excel表格
data_url = 'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls'
data_file_path = "./data/titanic3.xls"
if not os.path.isfile(data_file_path):
    result = urllib.request.urlretrieve(data_url, data_file_path)
    print('downloaded', result)
else:
    print(data_file_path, "data file already exists")

df_data = pd.read_excel(data_file_path)


def prepare_data(df_data):
    df = df_data.drop(['name'], axis=1)  # 删除姓名列
    age_mean = df['age'].mean()
    df['age'] = df['age'].fillna(age_mean)  # 为缺失age值的记录填充 age 平均值
    df['fare'] = df['fare'].fillna(df['fare'].mean())  # fare填充平均值
    df['sex'] = df['sex'].map({
        'female': 0,
        'male': 1
    }).astype(int)  # 将sex 字段转换为数值
    df['embarked'] = df['embarked'].fillna('S')
    df['embarked'] = df['embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)

    ndarray_data = df.values  # 转换为ndarray数组

    features = ndarray_data[:, 1:]  # 后七列是特征值
    label = ndarray_data[:, 0]  # 第0列是标签值

    # 特征值归一化
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    norm_features = minmax_scale.fit_transform(features)

    return norm_features, label


# 查看数据摘要
print(df_data.describe())

# 筛选提取需要的特征字段，去掉 ticke, cabin 等
# survived 是标签字段
selected_cols = [
    'survived', 'name', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
    'embarked'
]
selected_data = df_data[selected_cols]
# print(selected_data)

# # 把为空的字段输出为 true
# print(selected_data.isnull())
# # 将有null的列输出为 true
# print(selected_data.isnull().any())
# # 统计每一列的 null 值数量
# print(selected_data.isnull().sum())
# # 显示存在缺失值的行一句，确定缺失值的位置
# print(selected_data[selected_data.isnull().values == True])

# 打乱，在处理数据
x_data, y_data = prepare_data(selected_data.sample(frac=1))

# 划分训练集与测试集
train_size = int(len(x_data) * 0.8)

x_train = x_data[:train_size]
y_train = y_data[:train_size]

x_test = x_data[train_size:]
y_test = y_data[train_size:]

# 建立神经网络模型
model = tf.keras.models.Sequential()
# 加入第一层，输入特征数据 是7 列， 也可以用input_shape=(7,)
model.add(
    tf.keras.layers.Dense(units=64,
                          input_dim=7,
                          kernel_initializer='uniform',
                          bias_initializer='zeros',
                          activation='relu'))
# 第二层
model.add(tf.keras.layers.Dense(units=32, activation='sigmoid'))
# 输出层
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

model.summary()

# 模型设置
model.compile(optimizer=tf.keras.optimizers.Adam(0.003),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 设置回调
logdir = './logs'  # 模型训练日志的保存路径
checkpoint_path = './checkpoint/Titanic.{epoch:02d}-{val_loss:.2f}.ckpt'  # 检查点路径

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=2),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                       save_weights_only=True,
                                       verbose=1,
                                       period=5)
]

# 训练模型
train_history = model.fit(
    x=x_train,
    y=y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=40,
    verbose=2,
    callbacks=callbacks  # 设置回调
)

# 模型评估
evaluate_result = model.evaluate(x=x_test, y=y_test)
print(model.metrics_names, evaluate_result)

print(train_history.history.keys())

# 模型应用
Jack_info = [0, 'Jack', 3, 'male', 23, 1, 0, 5.000, 'S']
Rose_info = [1, 'Rose', 1, 'female', 20, 1, 0, 100.0000, 'S']

# 创建新的旅客 DataFrame
new_passenger_pd = pd.DataFrame([Jack_info, Rose_info], columns=selected_cols)
all_passenger_pd = selected_data.append(new_passenger_pd)

print(all_passenger_pd[-3:])

# 重新处理数据 ，使用model.predict() 进行预测
x_features, y_label = prepare_data(all_passenger_pd)
surv_probability = model.predict(x_features)
all_passenger_pd.insert(len(all_passenger_pd.columns), 'surv_probability',
                        surv_probability)
print(all_passenger_pd[-5:])


def visu_train_history(train_history, train_metric, validation_metric):
    plt.plot(train_history.history[train_metric])
    plt.plot(train_history.history[validation_metric])
    plt.title('Train History')
    plt.ylabel(train_metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# visu_train_history(train_history, 'accuracy', 'val_accuracy')
# visu_train_history(train_history, 'loss', "val_loss")