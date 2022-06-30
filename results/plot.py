import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
data = np.load(r'./MAE_history.npy')
x = np.arange(np.size(data,0))
for k in range(np.size(data,1)):
    plt.plot(x, data[:,k], '-o', label='Error of timestep {} prediction'.format(k+1))
plt.xlabel('training epoch')
plt.ylabel('Mean absolute error')
plt.legend(loc=0)
plt.grid()
plt.savefig(r'./MAE_history.png')