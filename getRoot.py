a = int(input("请输入二次项系数："))
b = int(input('请输入一次项系数：'))
c = int(input('请输常数项'))
rootList = []
delta = (b**2)-(4*a*c)
if delta>=0:
    for op in [1,-1]:
        rootList.append((-b+op*(delta**0.5))/(2*a))
    print(rootList)
else:
    print("无解")