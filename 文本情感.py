'''
非预设数据处理
如果是自有数据，或者网络其他数据，需要写一套处理数据的程序
基本思路：
    1、获取数据，确定数据格式规范；
    2、文字分词。英文分词可以按照空格分词，中文分词可以参考jieba；
    3、建立词索引表，给每个词一个数字索引编号；
    4、段落文字转为词索引向量；
    5、段度文字转为词嵌入矩阵；
'''

# 下载数据集

import os
import tarfile
import urllib.request
import tensorflow as tf
import numpy as np
import re
import string
from random import randint

# 数据地址
url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filepath = 'data/aclImdb_v1.tar.gz'

# 下载数据集
if not os.path.exists("data"):
    os.makedirs('data')

if not os.path.isfile(filepath):
    print('downloading...')
    result = urllib.request.urlretrieve(url, filepath)
    print('downloaded')
else:
    print(filepath, 'is existed')

if not os.path.exists('data/aclImdb'):
    tfile = tarfile.open(filepath, 'r:gz')
    print('extraciting...')
    result = tfile.extractall('data/')
    print('extraction completed')
else:
    print('data/aclImdb is existed!!')


# 清除html标签
def remove_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


# 读取文件
# filetype   =>    train  或  test
def read_files(filetype):
    path = './data/aclImdb/'
    file_list = []

    # 读取正面评价的文件的路径，存到file_list列表里
    positive_path = path + filetype + '/pos/'
    for f in os.listdir(positive_path):
        file_list += [positive_path + f]
    pos_file_num = len(file_list)

    # 读取负面评价的文件的路径， 存到file_list列表里
    negative_path = path + filetype + '/neg/'
    for f in os.listdir(negative_path):
        file_list += [negative_path + f]
    neg_files_num = len(file_list) - pos_file_num

    # 得到所有标签，用独热编码表示，正面[1,0],负面[0,1]
    all_labels = ([[1, 0]] * pos_file_num + [[0, 1]] * neg_files_num)

    # 得到所有文本
    all_texts = []
    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:
            all_texts += [remove_tags(''.join(file_input.readlines()))]

    return all_labels, all_texts


# 取出 训练 与 测试 的标签与文本
train_labels, train_texts = read_files('train')
test_labels, test_texts = read_files('test')

# 标签转为Tensor
train_labels = tf.constant(train_labels, dtype=tf.float32)
test_labels = tf.constant(test_labels, dtype=tf.float32)

# 建立词汇词典 Token
token = tf.keras.preprocessing.text.Tokenizer(num_words=4000)
token.fit_on_texts(train_texts)

# print('token 读取的文档数量：', token.document_count)
# print('单词的频率排名索引：', token.word_index, sep='\n')
# print('含有某单词的文本数量：', token.word_docs, sep='\n')
# print('单词出现次数：', token.word_counts, sep='\n')

# 将文字转为数字列表
train_sequences = token.texts_to_sequences(train_texts)
test_sequences = token.texts_to_sequences(test_texts)

# 文字与索引的对照
# print(train_texts[7])
# print(train_sequences[7])

x_train = tf.keras.preprocessing.sequence.pad_sequences(train_sequences,
                                                        padding='post',
                                                        truncating='post',
                                                        maxlen=400)
x_test = tf.keras.preprocessing.sequence.pad_sequences(test_sequences,
                                                       padding='post',
                                                       truncating='post',
                                                       maxlen=400)

print("x_train.shape", x_train.shape)

# 建立模型
model = tf.keras.models.Sequential()
# 嵌入
model.add(
    tf.keras.layers.Embedding(
        output_dim=32,  # 输出词向量的维度
        input_dim=4000,  # 输入的词汇表的长度
        input_length=400  # 输入Tensorr 的长度
    ))

# 平坦层
# model.add(tf.keras.layers.Flatten())

# 在使用 RNN 与 LSTM时 不需要平坦层
model.add(tf.keras.layers.LSTM(units=8))

# 全连接层
model.add(tf.keras.layers.Dense(units=256, activation='relu'))
# 防止过拟合
model.add(tf.keras.layers.Dropout(0.3))
# 输出层
model.add(tf.keras.layers.Dense(units=2, activation='softmax'))

# 设置模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 模型摘要
print(model.summary())

# 准备回调

logdir = './文本情感logs'  # 模型训练日志的保存路径
checkpoint_path = './checkpoint/文本情感.{epoch:02d}-{val_loss:.2f}.H5'  # 检查点路径

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=2),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                       save_weights_only=True,
                                       verbose=1,
                                       save_freq='epoch')
]
# 训练模型
history = model.fit(x=x_train,
                    y=train_labels,
                    validation_split=0.2,
                    epochs=50,
                    batch_size=128,
                    callbacks=callbacks,
                    verbose=2)

# 模型评估
testLoss, testAcc = model.evaluate(x_test, test_labels, verbose=1)
print("损失: ", testLoss, "准确率：", testAcc)
