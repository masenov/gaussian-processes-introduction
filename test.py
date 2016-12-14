from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import holoviews as hv
import numpy as np

interval = 10
interval_size = 0.01

mpl.rcParams.update({'font.size':14})
mpl.rcParams['xtick.major.pad']='8'
mpl.rcParams['ytick.major.pad']='8'
mpl.rcParams['lines.linewidth'] = 4
plt.rc('text', usetex=True)


def k(i,j,sigma_f,l):
    return np.power(sigma_f,2)*np.exp(-(np.power((i[0]-j[0]),2)/(2*np.power(l,2)) + np.power((i[1]-j[1]),2)/(2*np.power(l,2))))

def direct_delta(i,j):
    if (i==j):
        return 1
    else:
        return 0

def sigma(ndim=6,length=6,uniform=False,sigma_v=0.1,l=3,sigma_f=1):
    if (uniform):
        x = np.random.uniform(1,length,ndim)
        x.sort()
        y = np.random.uniform(1,length,ndim)
        y.sort()
    else:
        x = np.linspace(1, length, ndim)
        y = np.linspace(1, length, ndim)
    cov = np.zeros((ndim,ndim))
    for i in range(ndim):
        for j in range(ndim):
            cov[i,j]=k((x[i],y[i]),(x[j],y[j]),sigma_f,l)+np.power(sigma_v,2)*direct_delta(i,j)
    return cov,x,y



from matplotlib import cm

samples=1
ndim=50
cov,x,y = sigma(ndim,uniform=True)
mu=np.zeros(ndim)
z=np.random.multivariate_normal(mu, cov, samples).T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(x,y,np.squeeze(z),cmap=cm.jet)
fig.show()
