'''
生成一个[0,1)之间均匀分布的随机数数组，
包含1000个元素，
随机种子为612。
接收用户输入一个1-100之间的数字。
打印随机数组中所有索引值可以被输入整数整除的数字，并打印序号和索引值。
序号从1开始，依次加1.  
例如，用户输入50，则打印数组中索引值为0,50,100...1000的随机数。
'''
import numpy as np
np.random.seed(612)
ndArr = np.random.uniform(0,1,1000)
iNum = int(input("请输入一个1-100的整数:\n"))
while not(0<iNum<101):
    iNum = int(input("请重新输入一个1-100的整数:\n"))
k = 1;
for i in range(ndArr.size):
    if i%50 == 0:
        print(k,i,ndArr[i],sep="\t")
        k += 1


