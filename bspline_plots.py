import tt_iga
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

basis1 = tt_iga.bspline.BSplineBasis(np.array([0,0.25,0.5,0.75,1]),2)

x = np.linspace(0,1,1000)
y = basis1(x)
dy = basis1(x, derivative=True)
y[y==0] = np.nan
dy[dy==0] = np.nan

plt.figure()
plt.plot(x,y.T)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.xticks([0,1/4,2/4,3/4,1])
plt.yticks([0,2/4,1])
tikzplotlib.save('./bspline1.tex')

plt.figure()
plt.plot(x,dy.T)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.xticks([0,1/4,2/4,3/4,1])
plt.yticks([-8,-4,0,4,8])
tikzplotlib.save('./bsplineD.tex')