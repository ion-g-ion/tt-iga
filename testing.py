import tt_iga
import tt_iga.bspline
import torch as tn
import numpy as np
import torchtt as tntt
import matplotlib.pyplot as plt
import matplotlib.colors

tn.set_default_dtype(tn.float64)

r = 1.0
R = 1.5
h = -0.25 

points = tn.ones((3,3,3,3))
points[0,0,0,:] = tn.tensor([r,0,0])
points[1,0,0,:] = tn.tensor([(r+R)/2,0,0])
points[2,0,0,:] = tn.tensor([R,0,0])
points[0,0,1,:] = tn.tensor([r,0,r])
points[1,0,1,:] = tn.tensor([(r+R)/2,0,(r+R)/2])
points[2,0,1,:] = tn.tensor([R,0,R])
points[0,0,2,:] = tn.tensor([0,0,r])
points[1,0,2,:] = tn.tensor([0,0,(r+R)/2])
points[2,0,2,:] = tn.tensor([0,0,R])
points[:,1,:,:] = points[:,0,:,:]
points[:,1,:,1] = h/2
points[:,2,:,:] = points[:,0,:,:]
points[:,2,:,1] = h

points = [tntt.TT(points[...,0]), tntt.TT(points[...,1]), tntt.TT(points[...,2])]

weights = tn.ones((3,3,3))
weights[:,:,1] = 1/np.sqrt(2)
weights = tntt.TT(weights)

basis1 = tt_iga.bspline.BSplineBasis(np.linspace(-1,1,2),2)
basis2 = tt_iga.bspline.BSplineBasis(np.linspace(-1,1,2),2)
basis3 = tt_iga.bspline.BSplineBasis(np.linspace(-1,1,2),2)

geom = tt_iga.PatchNURBS([basis1,basis2,basis3], [], points, weights)


xs = geom([np.linspace(-1,1,32),np.linspace(-1,1,32),np.linspace(-1,1,32)])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xs[0].numpy(),xs[1].numpy(),xs[2].numpy(),s=1)

BasisSpace = [tt_iga.bspline.BSplineBasis(np.linspace(-1,1,31),2), tt_iga.bspline.BSplineBasis(np.linspace(-1,1,31),2), tt_iga.bspline.BSplineBasis(np.linspace(-1,1,31),2)]
N = [b.N for b in BasisSpace]

Mtt = geom.mass_interp(BasisSpace)
Stt = geom.stiffness_interp(BasisSpace, verb = False)

Pin_tt, Pbd_tt = tt_iga.projectors.get_projectors(N,[[1,1],[1,1],[0,0]])

f_tt = tntt.zeros(Stt.N)

# interpoalte the excitation and compute the correspinding tensor
u0 = 1
tmp = np.zeros(N)
tmp[:,:,-1] = u0
g_tt =  Pbd_tt @ tntt.TT(tmp) 

# assemble the system matrix
M_tt = Pin_tt@Stt@Pin_tt + Pbd_tt
rhs_tt = Pin_tt @ (Mtt @ f_tt - Stt @ Pbd_tt @ g_tt) + g_tt
M_tt = M_tt.round(1e-11)

# solve the system
eps_solver = 1e-7
dofs_tt = tntt.solvers.amen_solve(M_tt, rhs_tt, x0 = tntt.ones(rhs_tt.N), eps = eps_solver, nswp = 50, preconditioner = 'c',  verbose = False)

fspace = tt_iga.Function(BasisSpace)
fspace.dofs = dofs_tt

u_val = fspace([tn.linspace(0,1,1),tn.linspace(0,1,128),tn.linspace(0,1,128)]).full()
x,y,z = geom([tn.linspace(0,1,1),tn.linspace(0,1,128),tn.linspace(0,1,128)])

plt.figure()
ax = plt.axes(projection='3d')
C = u_val.numpy().squeeze()
norm = matplotlib.colors.Normalize(vmin=C.min(),vmax=C.max())
C = plt.cm.jet(norm(C))
C[:,:,-1] = 1
ax.plot_surface(x.numpy().squeeze(), y.numpy().squeeze(), z.numpy().squeeze(), edgecolors=None, linewidth=0, facecolors = C, antialiased=True, rcount=256, ccount=256, alpha=0.5)
plt.show()
