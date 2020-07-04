import tensorflow as tf
import numpy as np

# from matplotlib import pyplot as plt
	# from PIL import Image
	# src = "../Deep-Learning-Course/res/lena.tiff"
	# img = Image.open(src)
	# img_r, img_g, img_b = img.split()
	# ndImg_r = np.array(img_r)
	# tsImg_r = tf.constant(ndImg_r)
	# print(tsImg_r)
	# y = tf.split(tsImg_r,3,1)
	# print(y)
	# Image.open
	# plt.figure(figsize=(10, 10))
	# plt.rcParams['font.family'] = 'FZShuTi'
	# plt.imshow(img)
	# plt.show()

x = tf.constant(np.array(range(100)).reshape((10,10)))
print(x)

y = tf.gather(x,[3,6,7],axis = 1)
print(y)