import numpy as np
import pylab
from matplotlib import pyplot as plt

# N = 17
# a = np.random.rand(N,3)
# length = len(a)
# print(length)
# tenPercent = round(length/10)
# print(tenPercent)
# print(a)
# indices = np.random.permutation(a.shape[0])
# training_idx, test_idx = indices[:tenPercent], indices[length-tenPercent:]
# print(indices)
# print(training_idx)
# print(test_idx)
# front = np.take(a, training_idx, 0)
# end = np.take(a, test_idx, 0)
# print("~~~~~~~~")
# #front = a.take(training_idx)
# #end = a.take(test_idx)
# print(front)
# print(end)

# import matplotlib.pyplot as plt

#
# test = [1, 3, 6, 2, 3, 4, 6, 7, 7, 4, 5, 23, 52, 13, 24]
# test2 = [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13]
# pylab.plot(test, 'r', label='test')
# pylab.plot(test2, 'g', label='test2')
# pylab.legend(loc='upper left')
# pylab.xlabel('number of examples N')
# pylab.ylabel('number of mistakes M')
# pylab.show()
# plt.xlabel('number of examples N')
# plt.ylabel('number of mistakes M')
# plt.plot(test, 'r', label='test', test2, 'g', label='test2')
# plt.show()

# temp = np.ones(10)
# print(temp)
# temp2 = np.multiply(3, temp)
# print(temp2)
# temp3 = 3*temp
# print(temp3)
# root = np.sqrt(temp2)
# print(root)
# print(temp)

# for n in range(40, 200, 40):
#     print(n)

temp1 = [1,2,3,4,5]
temp2 = [2,4,6,8,10]
fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(temp1, temp2, 'r*', label = 'perceptron No margin')
plt.legend()

for xy in zip(temp1, temp2):                                       # <--
    print(xy)
    print(type(xy))
    ax.annotate('(n=%s, #mistake=%s)' % (temp1[0], temp2[0]), xy=xy, textcoords='data')         # <--


plt.show()