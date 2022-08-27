import torchtt as tntt
import numpy as np
import tt_iga
import matplotlib.pyplot as plt
import torch as tn
import tikzplotlib

tn.set_default_dtype(tn.float64)


knots1 = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[2.0,1.0],[2.0,2.0]])
weights1 = np.array([1.0,1,0.6,1,1])
basis1 = tt_iga.bspline.BSplineBasis(np.array([0,0.375,0.625,1]),2)

geom1 = tt_iga.PatchNURBS([basis1],[], [tntt.TT(knots1[:,0]), tntt.TT(knots1[:,1])], tntt.TT(weights1))

y = tn.linspace(0,1,128)
x1, x2 = geom1([y])

plt.figure()
plt.plot(knots1[:,0],knots1[:,1],'k:',linewidth=1)
plt.plot(x1.numpy(),x2.numpy())
plt.scatter(knots1[:,0],knots1[:,1],c='k')
plt.grid()
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
tikzplotlib.save('nurbs_curve_2.tex')


knots2 = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[1.0,1.0],[2.0,1.0],[2.0,2.0]])
weights2 = np.array([1.0,1,0.6,1,1,1])
basis2 = tt_iga.bspline.BSplineBasis(np.array([0,0.375,0.5,0.625,1]),2)

geom2 = tt_iga.PatchNURBS([basis2],[], [tntt.TT(knots2[:,0]), tntt.TT(knots2[:,1])], tntt.TT(weights2))

y = tn.linspace(0,1,128)
x1, x2 = geom2([y])

plt.figure()
plt.plot(knots2[:,0],knots2[:,1],'k:',linewidth=1)
plt.plot(x1.numpy(),x2.numpy())
plt.scatter(knots2[:,0],knots2[:,1],c='k')
plt.grid()
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
tikzplotlib.save('nurbs_curve_3.tex')

knots3 = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[2.0,1.0],[2.0,2.0]])
weights3 = np.array([1.0,1,0.6,1,1])
basis3 = tt_iga.bspline.BSplineBasis(np.linspace(0,1,5),1)

geom3 = tt_iga.PatchNURBS([basis3],[], [tntt.TT(knots3[:,0]), tntt.TT(knots3[:,1])], tntt.TT(weights3))

y = tn.linspace(0,1,128)
x1, x2 = geom3([y])

plt.figure()
plt.plot(knots3[:,0],knots3[:,1],'k:',linewidth=1)
plt.plot(x1.numpy(),x2.numpy())
plt.scatter(knots3[:,0],knots3[:,1],c='k')
plt.grid()
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
tikzplotlib.save('nurbs_curve_1.tex')