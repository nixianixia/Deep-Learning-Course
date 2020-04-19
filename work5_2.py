import numpy as np
x = [64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03]
y = [62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84]

ndx = np.array(x)
ndy = np.array(y)

ndxMean = ndx.mean()
ndyMean = ndy.mean()
fz = 0.0
fm = 0.0
for i in range(ndx.size):
    ndx[i] -= ndxMean
    ndy[i] -= ndyMean
    fz += ndx[i] * ndy[i]
    fm += ndx[i] ** 2

w = fz/fm
b = ndyMean - w*ndxMean

print("w = ",w)
print("b = ",b)
