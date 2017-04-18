import matplotlib.pyplot as plt
import numpy as np

loss0 = np.load('output/loss0.npy')
loss1 = np.load('output/loss1.npy')
loss2 = np.load('output/loss2.npy')
loss3 = np.load('output/loss3.npy')

data0 = []
data1 = []
data2 = []
data3 = []

for i in range(0, 20, 1):
    data0.append(loss0[i])
    data1.append(loss1[i])
    data2.append(loss2[i])
    data3.append(loss3[i])

x = []
for i in range(0, 20, 1):
    x.append(i / 1100)

plt.figure(1)
curve0, = plt.semilogy(x, data0, 'r')
curve1, = plt.semilogy(x, data1, 'b')

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right', handles=[curve0, curve1], labels=['Ada', 'Ada_Momentum'])
plt.grid(True)
plt.show()

plt.figure(2)
curve2, = plt.semilogy(x, data2, 'r')
curve3, = plt.semilogy(x, data3, 'b')

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right', handles=[curve2, curve3], labels=['RMS', 'RMS_Mo'])
plt.grid(True)
plt.show()

# plt.figure(1)
# curve0, = plt.plot(loss0, 'r')
# curve1, = plt.plot(loss1, 'b')
#
#
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(loc='upper right', handles=[curve0, curve1], labels=['Ada', 'Ada_Mo', 'RMS', 'RMS_Mo'])
# plt.show()

